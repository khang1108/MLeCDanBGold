"""Audio extraction and normalization performed before ASR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import pandas as pd
from offline.enrichment.transcripts.models import TranscriptSegment
from hcmai.common.utils.io import atomic_write, write_json
from offline.enrichment.transcripts.adapters.asr import ASRAdapter, read_audio
from offline.enrichment.transcripts.adapters.diarization import DiarizationAdapter
from offline.enrichment.transcripts.adapters.remote import (
    RemoteASRAdapter,
    RemoteDiarizationAdapter,
)

from offline.enrichment.transcripts.manifest import (
    TranscriptManifest,
    expected_manifest,
    failure_manifest,
    fingerprint_source,
    reusable_transcript,
)

from offline.enrichment.transcripts.publication import publish_staged, staging_path
from offline.enrichment.transcripts.artifacts import (
    load_transcript_artifact_records,
)
from offline.ingestion.s3 import VIDEO_EXTENSIONS
from tqdm import tqdm

TRANSCRIPT_DTYPES = {
    "segment_id": "string",
    "video_id": "string",
    "segment_index": "int64",
    "start_ms": "int64",
    "end_ms": "int64",
    "text": "string",
    "language": "string",
    "speaker_id": "string",
    "confidence": "Float64",
    "status": "string",
    "model_name": "string",
    "model_revision": "string",
    "artifact_version": "string",
    "error_code": "string",
    "error_message": "string",
}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptReport:
    """Summary of one transcript preparation run."""

    expected: int
    transcribed: int
    no_speech: int
    failed: dict[str, str]
    segments: int
    output_path: Path


def _table(records: list[TranscriptSegment]) -> pd.DataFrame:
    """Convert transcript records to a table with stable nullable types."""

    table = (
        pd.DataFrame(
            [record.model_dump(mode="json") for record in records],
            columns=list(TRANSCRIPT_DTYPES),
        )
        if records
        else pd.DataFrame(
            {name: pd.Series(dtype=dtype) for name, dtype in TRANSCRIPT_DTYPES.items()}
    ))
    return table.astype(TRANSCRIPT_DTYPES)


def _validate_records(records: list[TranscriptSegment], video_id: str) -> list[TranscriptSegment]:
    """Validate canonical identity, order, and monotonic media intervals."""

    expected_indexes = list(range(len(records)))
    if [record.segment_index for record in records] != expected_indexes:
        raise ValueError("transcript segment indexes must be consecutive")
    if any(record.video_id != video_id for record in records):
        raise ValueError("transcript provider changed canonical video identity")
    if len({record.segment_id for record in records}) != len(records):
        raise ValueError("transcript segment IDs must be unique")
    if any(record.start_ms < 0 or record.end_ms <= record.start_ms for record in records):
        raise ValueError("transcript intervals must have positive media duration")
    for previous, current in zip(records, records[1:]):
        if current.start_ms < previous.start_ms or current.end_ms < previous.end_ms:
            raise ValueError("transcript intervals must be monotonic")
    return records


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _failure_path(output: Path) -> Path:
    return output.with_suffix(".failure.json")


def _load_one(output: Path) -> list[TranscriptSegment]:
    return list(load_transcript_artifact_records(output))


def _write_validated_pair(
    records: list[TranscriptSegment],
    manifest: TranscriptManifest,
    output: Path,
) -> None:
    """Stage, reread, validate, and recoverably promote Parquet plus manifest."""

    manifest_path = _manifest_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged_output = staging_path(output)
    staged_manifest = staging_path(manifest_path)
    try:
        _table(records).to_parquet(staged_output, index=False)
        write_json(manifest.model_dump(mode="json"), staged_manifest)
        staged_records = _validate_records(_load_one(staged_output), manifest.video_id)
        staged_manifest_value = TranscriptManifest.model_validate_json(
            staged_manifest.read_text(encoding="utf-8")
        )
        if staged_manifest_value != manifest:
            raise ValueError("staged transcript manifest changed during serialization")
        if len(staged_records) != manifest.segment_count:
            raise ValueError("staged transcript count does not match manifest")
        publish_staged({
            output: staged_output,
            manifest_path: staged_manifest,
        })
    finally:
        staged_output.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)


def _prepare_video(
    engine: ASRAdapter | RemoteASRAdapter,
    diarizer: DiarizationAdapter | RemoteDiarizationAdapter | None,
    video: Path,
    output: Path,
    *,
    resume: bool,
    schema_version: str,
    pipeline_version: str,
) -> tuple[Path, int]:
    """Prepare or safely reuse one transcript/manifest pair."""

    expected = expected_manifest(
        video,
        video.stem,
        engine.config,
        diarizer.config if diarizer is not None else None,
        asr_revision=engine.resolved_revision,
        diarization_revision=(diarizer.resolved_revision if diarizer is not None else None),
        schema_version=schema_version,
        pipeline_version=pipeline_version,
    )
    output.with_suffix(f"{output.suffix}.partial").unlink(missing_ok=True)
    if resume and output.exists():
        try:
            records = _validate_records(_load_one(output), video.stem)
        except Exception:
            records = None
        if records is not None and reusable_transcript(
            output, _manifest_path(output), expected, records
        ):
            logger.info(
                "Transcript checkpoint reused: video=%s segments=%d",
                video.stem,
                len(records),
            )
            return output, len(records)

    try:
        started = perf_counter()
        decode_seconds = 0.0
        transcribe_audio_fn = getattr(engine, "transcribe_audio", None)
        decoded_api = callable(transcribe_audio_fn) and (
            diarizer is None or callable(getattr(diarizer, "assign_speakers_audio", None))
        )
        if decoded_api and callable(transcribe_audio_fn):
            decode_started = perf_counter()
            decoded = read_audio(video, engine.config.audio_sample_rate)
            decode_seconds = perf_counter() - decode_started
            asr_started = perf_counter()
            records = cast(
                list[TranscriptSegment],
                transcribe_audio_fn(decoded, video.stem),
            )
        else:
            decoded = None
            asr_started = perf_counter()
            records = engine.transcribe(video, video.stem)
        asr_seconds = perf_counter() - asr_started
        if engine.resolved_revision != engine.config.revision:
            raise ValueError("ASR backend resolved a revision different from its pin")
        if diarizer is not None:
            diarization_started = perf_counter()
            if decoded is None:
                records = diarizer.assign_speakers(video, records)
            else:
                if diarizer.config.audio_sample_rate != decoded.sample_rate:
                    raise ValueError("ASR and diarization must use the same audio sample rate")
                records = diarizer.assign_speakers_audio(decoded, records)
            diarization_seconds = perf_counter() - diarization_started
        else:
            diarization_seconds = 0.0
        records = _validate_records(records, video.stem)
        completed = expected.model_copy(update={"segment_count": len(records)})
        _write_validated_pair(records, completed, output)
        _failure_path(output).unlink(missing_ok=True)
        logger.info(
            "Transcript preparation completed: video=%s segments=%d "
            "audio_decode_seconds=%.1f asr_seconds=%.1f "
            "diarization_seconds=%.1f total_seconds=%.1f",
            video.stem,
            len(records),
            decode_seconds,
            asr_seconds,
            diarization_seconds,
            perf_counter() - started,
        )
        return output, len(records)
    except Exception as error:
        failed = failure_manifest(expected, error)
        atomic_write(
            _failure_path(output),
            lambda path: write_json(failed.model_dump(mode="json"), path),
        )
        raise


def prepare_transcript_video(
    video_path: str | Path,
    output_root: str | Path,
    engine: ASRAdapter | RemoteASRAdapter,
    *,
    diarizer: DiarizationAdapter | RemoteDiarizationAdapter | None = None,
    resume: bool = True,
    schema_version: str = "transcript-segment-v1",
    pipeline_version: str = "transcript-pipeline-v1",
) -> tuple[Path, int]:
    """Prepare one staged video without rediscovering or copying its source."""

    video = Path(video_path).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"Supported staged video does not exist: {video}")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return _prepare_video(
        video=video,
        output=_video_output(root, video.stem),
        engine=engine,
        diarizer=diarizer,
        resume=resume,
        schema_version=schema_version,
        pipeline_version=pipeline_version,
    )


def _video_files(root: Path, limit: int | None) -> list[Path]:
    """Find supported videos in deterministic canonical-ID order."""

    if not root.is_dir():
        raise FileNotFoundError(f"Videos root does not exist: {root}")
    video_roots = sorted(root.glob("Videos_*/video")) or [root]
    candidates = sorted(
        path.resolve()
        for video_root in video_roots
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    videos: dict[str, Path] = {}
    for path in candidates:
        current = videos.get(path.stem)
        if current and fingerprint_source(current) != fingerprint_source(path):
            raise ValueError(f"Conflicting video_id with different source content: {path.stem}")
        videos.setdefault(path.stem, path)
    return [videos[video_id] for video_id in sorted(videos)][:limit]


def _video_output(root: Path, video_id: str) -> Path:
    """Return the grouped transcript path without deriving canonical frame IDs."""

    group = video_id.split("_", maxsplit=1)[0]
    return root / group / f"{video_id}.parquet"


def prepare_transcripts(
    videos_root: str | Path,
    output_path: str | Path,
    engine: ASRAdapter | RemoteASRAdapter,
    *,
    diarizer: DiarizationAdapter | RemoteDiarizationAdapter | None = None,
    resume: bool = True,
    limit: int | None = None,
    schema_version: str = "transcript-segment-v1",
    pipeline_version: str = "transcript-pipeline-v1",
) -> TranscriptReport:
    """Write independently resumable transcript artifacts without cross-video loss."""

    root = Path(videos_root).expanduser().resolve()
    output_root = Path(output_path).expanduser().resolve()
    videos = _video_files(root, limit)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    segment_counts: list[int] = []
    for video in tqdm(videos, desc="ASR transcript preparation", unit="video"):
        try:
            _, count = prepare_transcript_video(
                video_path=video,
                output_root=output_root,
                engine=engine,
                diarizer=diarizer,
                resume=resume,
                schema_version=schema_version,
                pipeline_version=pipeline_version,
            )
            segment_counts.append(count)
        except Exception as error:
            failures[video.stem] = type(error).__name__
    return TranscriptReport(
        expected=len(videos),
        transcribed=sum(count > 0 for count in segment_counts),
        no_speech=sum(count == 0 for count in segment_counts),
        failed=failures,
        segments=sum(segment_counts),
        output_path=output_root,
    )
