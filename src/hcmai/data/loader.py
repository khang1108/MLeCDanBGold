"""In-memory access to canonical frame metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.search import SearchFilters


class FrameStore:
    """Load frame metadata once and provide indexed access to its records."""

    def __init__(self, metadata_path: str | Path) -> None:
        """Load and index frame records from a Parquet file."""

        self.metadata_path = Path(metadata_path)
        table = pd.read_parquet(self.metadata_path)
        self._records = tuple(
            self._record_from_row(row)
            for row in table.to_dict(orient="records")
        )
        self._records_by_id = {
            record.frame_id: record for record in self._records
        }

        if len(self._records_by_id) != len(self._records):
            raise ValueError(
                f"Duplicate frame_id values in {self.metadata_path}"
            )

        records_by_video: defaultdict[str, list[FrameRecord]] = defaultdict(
            list
        )
        for record in self._records:
            records_by_video[record.video_id].append(record)
        self._records_by_video = {
            video_id: tuple(
                sorted(
                    records,
                    key=lambda record: (
                        record.timestamp_ms,
                        record.frame_idx,
                        record.frame_id,
                    ),
                )
            )
            for video_id, records in records_by_video.items()
        }

    @staticmethod
    def _record_from_row(row: dict[str, object]) -> FrameRecord:
        """Validate one Parquet row against the canonical frame contract."""

        values = {
            name: row[name]
            for name in FrameRecord.model_fields
            if name in row
        }
        for name in ("thumbnail_path", "shot_id"):
            if name in values and pd.isna(values[name]):
                values[name] = None
        return FrameRecord.model_validate(values)

    def get(self, frame_id: str) -> FrameRecord:
        """Return one frame or raise a contextual error for an unknown ID."""

        try:
            return self._records_by_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.metadata_path}"
            ) from None

    def get_many(self, frame_ids: Sequence[str]) -> list[FrameRecord]:
        """Return frames in input order while preserving duplicate IDs."""

        return [self.get(frame_id) for frame_id in frame_ids]

    def get_neighbors(
        self,
        frame_id: str,
        *,
        window_ms: int,
        include_self: bool = False,
    ) -> list[FrameRecord]:
        """Return same-video frames within an inclusive temporal window."""

        if window_ms < 0:
            raise ValueError("window_ms must be greater than or equal to zero")

        frame = self.get(frame_id)
        start_time = frame.timestamp_ms - window_ms
        end_time = frame.timestamp_ms + window_ms
        return [
            neighbor
            for neighbor in self._records_by_video[frame.video_id]
            if start_time <= neighbor.timestamp_ms <= end_time
            and (include_self or neighbor.frame_id != frame.frame_id)
        ]

    def filter_frame_ids(
        self,
        filters: SearchFilters | None,
    ) -> list[str]:
        """Return IDs matching video and inclusive time filters."""

        if filters is None:
            return [record.frame_id for record in self._records]

        video_ids = set(filters.video_ids)
        return [
            record.frame_id
            for record in self._records
            if (not video_ids or record.video_id in video_ids)
            and (
                filters.start_time_ms is None
                or record.timestamp_ms >= filters.start_time_ms
            )
            and (
                filters.end_time_ms is None
                or record.timestamp_ms <= filters.end_time_ms
            )
        ]
