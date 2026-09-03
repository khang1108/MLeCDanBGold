"""Load validated segment-native transcript evidence for timeline lookup.

The store preserves segment timing and provenance from Parquet. It does not
align speech to frames or depend on deterministic FrameContext generation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hcmai.corpus.models import TranscriptSegment


class _TranscriptArtifact(BaseModel):
    """Validate canonical timeline fields at the runtime artifact boundary."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    segment_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        """Require every transcript artifact segment to have positive duration."""

        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


def load_transcript_records(
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
        rows = table.where(table.notna(), cast(Any, None)).to_dict(orient="records")
        for row in rows:
            artifact_record = _TranscriptArtifact.model_validate(row)
            records.append(
                TranscriptSegment(
                    segment_id=artifact_record.segment_id,
                    video_id=artifact_record.video_id,
                    segment_index=artifact_record.segment_index,
                    start_ms=artifact_record.start_ms,
                    end_ms=artifact_record.end_ms,
                    text=artifact_record.text,
                )
            )

    identifiers = [record.segment_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate segment_id values in {metadata_path}")
    return tuple(records)


class TranscriptStore:
    """Load transcript timeline evidence once for ID and temporal lookups."""

    def __init__(self, metadata_path: str | Path) -> None:
        """Load and index one transcript file or directory."""

        self.metadata_path = Path(metadata_path)
        records = load_transcript_records(self.metadata_path)
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

    def iter_records(self) -> Iterator[TranscriptSegment]:
        """Iterate every validated segment in deterministic artifact order."""

        return iter(self._records_by_id.values())

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
