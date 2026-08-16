"""Cụ thể hóa (Materialize) dữ liệu Transcript.

Đồng bộ và căn chỉnh kết quả ASR (nhận diện giọng nói) với các khung hình cụ thể để truy xuất.

Các tính năng chính:
1. Frame Alignment: So khớp khoảng thời gian của câu thoại (start/end) với timestamp của frames.
2. Phân tách (Chunking): Chia nhỏ các đoạn thoại dài thành các câu/từ phù hợp với độ dài cảnh.
3. Chuẩn hoá Metadata: Tạo ra bản ghi text-to-frame chuẩn để nạp vào hệ thống Evidence Store."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

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
from hcmai.data.enrichment.transcripts.manifest import load_manifest
from hcmai.data.enrichment.transcripts.store import TranscriptStore
from hcmai.data.stores.frame import FrameStore


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
    """Align half-open transcript intervals to evaluated canonical frames."""

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
    canonical_order: list[str],
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
    validate_frame_enrichment(rows_dict, canonical_order, frame_store_id)
    
    table = pd.DataFrame(
        [row.model_dump(mode="json") for row in rows],
        columns=list(FrameEnrichment.model_fields),
    )

    def write_and_validate(staging: Path) -> None:
        table.to_parquet(staging, index=False)
        values = pd.read_parquet(staging).astype(object)
        restored = [
            FrameEnrichment.model_validate({
                key: (None if pd.isna(value) else value)
                for key, value in record.items()
                if key != "objects"
            } | {
                "objects": (
                    record.get("objects").tolist()
                    if callable(getattr(record.get("objects"), "tolist", None))
                    else record.get("objects") or []
                )
            })
            for record in values.where(values.notna(), None).to_dict(orient="records")
        ]
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
    """Load canonical offline inputs and publish the online ASR evidence table."""

    frame_store = FrameStore(frames_path)
    transcript_path = Path(transcript_root)
    transcript_store = TranscriptStore(transcript_path)
    frames = list(frame_store.iter_frames())
    rows = materialize_asr_enrichment(
        frames,
        list(transcript_store.iter_records()),
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
