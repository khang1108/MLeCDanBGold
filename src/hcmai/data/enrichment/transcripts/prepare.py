"""Build one transcript Parquet per video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hcmai.common.schemas import TranscriptSegment
from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter
from hcmai.data.enrichment.transcripts.adapters.diarization import (
    DiarizationAdapter,
)

TRANSCRIPT_DTYPES = {
    "segment_id": "string",
    "video_id": "string",
    "segment_index": "int64",
    "start_ms": "int64",
    "end_ms": "int64",
    "text": "string",
    "language": "string",
    "speaker_id": "string",
}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True)
class TranscriptReport:
    """Summary of one transcript preparation run."""

    expected: int
    transcribed: int
    no_speech: int
    failed: dict[str, str]
    segments: int
    output_path: Path


def _table(
    records: list[TranscriptSegment],
) -> pd.DataFrame:
    """Convert transcript records to a table with stable types."""

    table = (
        pd.DataFrame(
            [record.model_dump(mode="python") for record in records],
            columns=list(TRANSCRIPT_DTYPES),
        )
        if records
        else pd.DataFrame({
            name: pd.Series(dtype=dtype)
            for name, dtype in TRANSCRIPT_DTYPES.items()
        })
    )
    return table.astype(TRANSCRIPT_DTYPES)


def _write_parquet(table: pd.DataFrame, path: Path) -> None:
    """Publish Parquet only after its temporary file is complete."""

    partial = path.with_suffix(f"{path.suffix}.partial")
    table.to_parquet(partial, index=False)
    partial.replace(path)


def _prepare_video(
    engine: ASRAdapter, diarizer: DiarizationAdapter,
    video: Path, output: Path,
) -> None:
    """Write one speaker-labelled transcript output."""

    records = engine.transcribe(video, video.stem)
    records = diarizer.assign_speakers(video, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(_table(records), output)


def _video_files(root: Path, limit: int | None) -> list[Path]:
    """Find supported videos in deterministic order."""

    if not root.is_dir():
        raise FileNotFoundError(f"Videos root does not exist: {root}")
    video_roots = sorted(root.glob("Videos_*/video")) or [root]
    candidates = sorted(
        path.resolve()
        for video_root in video_roots
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    videos: dict[str, Path] = {}
    for path in candidates:
        current = videos.get(path.stem)
        if current and current.stat().st_size != path.stat().st_size:
            raise ValueError(f"Conflicting video_id: {path.stem}")
        videos.setdefault(path.stem, path)
    return [videos[video_id] for video_id in sorted(videos)][:limit]


def _video_output(root: Path, video_id: str) -> Path:
    """Return the grouped output path for one video."""

    group = video_id.split("_", maxsplit=1)[0]
    return root / group / f"{video_id}.parquet"


def _count_outputs(paths: list[Path]) -> tuple[int, int, int]:
    """Count completed videos and transcript segments."""

    transcribed = 0
    segments = 0
    for path in paths:
        count = len(pd.read_parquet(path, columns=["segment_id"]))
        transcribed += int(count > 0)
        segments += count
    return transcribed, len(paths) - transcribed, segments


def _process_video(
    video: Path,
    output_root: Path,
    engine: ASRAdapter,
    diarizer: DiarizationAdapter,
    resume: bool,
) -> Path:
    """Prepare one transcript artifact for a video."""

    output = _video_output(output_root, video.stem)
    if not resume:
        output.unlink(missing_ok=True)
    if not output.exists():
        _prepare_video(engine, diarizer, video, output)
    return output


def prepare_transcripts(
    videos_root: str | Path, output_path: str | Path, engine: ASRAdapter,
    *, diarizer: DiarizationAdapter, resume: bool = True,
    limit: int | None = None,
) -> TranscriptReport:
    """Write resumable speaker-labelled transcripts for each video."""

    root = Path(videos_root).expanduser().resolve()
    output_root = Path(output_path).expanduser().resolve()
    videos = _video_files(root, limit)
    output_root.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    completed: list[Path] = []
    for video in videos:
        try:
            completed.append(_process_video(
                video, output_root, engine, diarizer, resume,
            ))
        except Exception as error:
            failures[video.stem] = str(error)
    transcribed, no_speech, segments = _count_outputs(completed)
    return TranscriptReport(
        len(videos), transcribed, no_speech, failures, segments,
        output_root,
    )
