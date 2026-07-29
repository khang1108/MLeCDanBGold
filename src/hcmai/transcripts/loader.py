"""In-memory access to canonical transcript metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hcmai.common.schemas import TranscriptSegment


def _records(
    metadata_path: Path,
) -> tuple[TranscriptSegment, ...]:
    """Load transcript records from one Parquet or a directory."""

    paths = (
        sorted(metadata_path.rglob("*.parquet"))
        if metadata_path.is_dir()
        else [metadata_path]
    )
    records = []
    for path in paths:
        table = pd.read_parquet(path).astype(object)
        rows = table.where(table.notna(), None).to_dict(orient="records")
        records.extend(TranscriptSegment.model_validate(row) for row in rows)
    return tuple(records)


class TranscriptStore:
    """Load transcript metadata once for ID and temporal lookups."""

    def __init__(self, metadata_path: str | Path) -> None:
        """Load and index one transcript file or directory."""

        self.metadata_path = Path(metadata_path)
        records = _records(self.metadata_path)
        self._records_by_id = {
            record.segment_id: record for record in records
        }
        by_video: defaultdict[str, list[TranscriptSegment]] = defaultdict(list)
        for record in records:
            by_video[record.video_id].append(record)
        self._records_by_video = {
            video_id: tuple(
                sorted(items, key=lambda item: item.segment_index)
            )
            for video_id, items in by_video.items()
        }

    def get(self, segment_id: str) -> TranscriptSegment:
        """Return one segment by its unique identifier."""

        return self._records_by_id[segment_id]

    def get_many(
        self,
        segment_ids: Sequence[str],
    ) -> list[TranscriptSegment]:
        """Return segments in input order while preserving duplicates."""

        return [self.get(segment_id) for segment_id in segment_ids]

    def get_by_video(self, video_id: str) -> list[TranscriptSegment]:
        """Return all segments for one video in segment order."""

        return list(self._records_by_video.get(video_id, ()))

    def get_at_time(
        self,
        video_id: str,
        timestamp_ms: int,
    ) -> list[TranscriptSegment]:
        """Return segments containing a half-open timestamp."""

        return [
            record
            for record in self._records_by_video.get(video_id, ())
            if record.start_ms <= timestamp_ms < record.end_ms
        ]

    def get_in_range(
        self,
        video_id: str,
        start_ms: int,
        end_ms: int,
    ) -> list[TranscriptSegment]:
        """Return segments overlapping the requested half-open range."""

        return [
            record
            for record in self._records_by_video.get(video_id, ())
            if record.start_ms < end_ms and record.end_ms > start_ms
        ]
