"""Derive the legacy frame-aligned ASR compatibility view.

Transcript segments are the ASR source of truth. This module only projects
their half-open timeline intervals onto canonical frames for consumers that
still require ``FrameEnrichment``; it does not feed or depend on FrameContext.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from hcmai.common.schemas import (
    FrameEnrichment,
    FrameRecord,
    ProcessingStatus,
    TranscriptSegment,
    validate_frame_enrichment,
)
from hcmai.common.utils.io import atomic_write
from offline.enrichment.transcripts.artifacts import (
    load_transcript_artifact_records,
)
from offline.enrichment.transcripts.manifest import load_manifest
from hcmai.corpus.stores.frame import FrameStore


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _deduplicated_text(segments: Iterable[TranscriptSegment]) -> str | None:
    seen: set[str] = set()
    values: list[str] = []
    for segment in segments:
        value = _normalize(segment.text)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            values.append(value)
    return " ".join(values) or None


def materialize_asr_enrichment(
    frames: Sequence[FrameRecord],
    segments: Sequence[TranscriptSegment],
    *,
    evaluated_video_ids: set[str],
    window_ms: int,
    enrichment_version: str,
    model_name: str,
    frame_store_id: str | None = None,
) -> list[FrameEnrichment]:
    """Derive a compatibility view aligned to evaluated canonical frames."""

    if window_ms < 0:
        raise ValueError("frame evidence window must be non-negative")
    canonical_videos = {frame.video_id for frame in frames}
    unknown_videos = (
        evaluated_video_ids | {segment.video_id for segment in segments}
    ) - canonical_videos
    if unknown_videos:
        raise ValueError(
            "Transcript data references unknown canonical videos: "
            + ", ".join(sorted(unknown_videos))
        )
    by_video: defaultdict[str, list[TranscriptSegment]] = defaultdict(list)
    for segment in segments:
        # A derived completed frame row must never turn partial or failed ASR
        # output into positive retrieval evidence.
        if segment.status == ProcessingStatus.COMPLETED:
            by_video[segment.video_id].append(segment)
    for values in by_video.values():
        values.sort(
            key=lambda segment: (
                segment.start_ms,
                segment.end_ms,
                segment.segment_index,
                segment.segment_id,
            )
        )

    rows: list[FrameEnrichment] = []
    for frame in tqdm(frames, desc="Aligning ASR to frames", unit="frame"):
        if frame.video_id not in evaluated_video_ids:
            continue
        start_ms = max(0, frame.timestamp_ms - window_ms)
        end_ms = frame.timestamp_ms + window_ms + 1
        overlapping = [
            segment
            for segment in by_video.get(frame.video_id, ())
            if segment.start_ms < end_ms and segment.end_ms > start_ms
        ]
        rows.append(FrameEnrichment(
            frame_id=frame.frame_id,
            frame_store_id=frame_store_id,
            asr_text=_deduplicated_text(overlapping),
            source_segment_ids=[segment.segment_id for segment in overlapping],
            enrichment_version=enrichment_version,
            model_name=model_name,
            status=ProcessingStatus.COMPLETED,
        ))
    return rows


def completed_video_ids(transcript_root: Path) -> set[str]:
    """Return explicitly evaluated videos, including completed no-speech videos."""

    completed: set[str] = set()
    for path in sorted(transcript_root.rglob("*.manifest.json")):
        manifest = load_manifest(path)
        if manifest.status == "completed":
            completed.add(manifest.video_id)
    return completed


def write_asr_enrichment(
    output_path: Path,
    rows: Sequence[FrameEnrichment],
    *,
    canonical_frame_ids: set[str],
    canonical_order: list[str] | None = None,
    frame_store_id: str | None = None,
) -> Path:
    """Validate every foreign key and atomically replace one online artifact."""

    identifiers = [row.frame_id for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ASR enrichment frame IDs must be unique")
    unknown = set(identifiers) - canonical_frame_ids
    if unknown:
        raise ValueError(
            "ASR enrichment references unknown frame IDs: "
            + ", ".join(sorted(unknown))
        )
    rows_dict = {row.frame_id: row for row in rows}
    validate_frame_enrichment(
        rows_dict,
        canonical_order or identifiers,
        frame_store_id,
    )
    
    table = pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=list(FrameEnrichment.model_fields),
    )

    def restore_value(value: object) -> Any:
        if value is None:
            return None
        if callable(getattr(value, "tolist", None)):
            return value.tolist()  # type: ignore[union-attr]
        if isinstance(value, (list, tuple)):
            return list(value)
        missing = pd.isna(value)
        return None if bool(missing) else value

    def write_and_validate(staging: Path) -> None:
        table.to_parquet(staging, index=False)
        values = pd.read_parquet(staging).astype(object)
        restored = []
        for record in values.where(values.notna(), None).to_dict(
            orient="records"
        ):
            raw_objects = restore_value(record.get("objects"))
            objects = list(raw_objects) if raw_objects is not None else []
            restored.append(FrameEnrichment.model_validate({
                key: restore_value(value)
                for key, value in record.items()
                if key != "objects"
            } | {"objects": objects}))
        if [row.frame_id for row in restored] != identifiers:
            raise ValueError("staged ASR enrichment changed canonical frame order")

    atomic_write(output_path, write_and_validate)
    return output_path


def materialize_transcript_artifact(
    frames_path: str | Path,
    transcript_root: str | Path,
    output_path: str | Path,
    *,
    window_ms: int,
    enrichment_version: str,
    model_name: str,
    frame_store_id: str | None = None,
) -> Path:
    """Publish a derived frame-aligned view from canonical transcript segments."""

    frame_store = FrameStore(frames_path)
    transcript_path = Path(transcript_root)
    frames = list(frame_store.iter_frames())
    rows = materialize_asr_enrichment(
        frames,
        load_transcript_artifact_records(transcript_path),
        evaluated_video_ids=completed_video_ids(transcript_path),
        window_ms=window_ms,
        enrichment_version=enrichment_version,
        model_name=model_name,
        frame_store_id=frame_store_id,
    )
    return write_asr_enrichment(
        Path(output_path),
        rows,
        canonical_frame_ids={frame.frame_id for frame in frames},
        canonical_order=[frame.frame_id for frame in frames],
        frame_store_id=frame_store_id,
    )
