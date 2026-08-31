"""In-memory access to canonical frame metadata.

This module provides ``FrameStore``, the single interface other pipeline
components use to look up frame records at runtime.  It is designed to
be instantiated **once at application startup** and kept alive for the
entire process lifetime.

Typical usage::

    from hcmai.corpus.stores.frame import FrameStore

    store = FrameStore("artifacts/frame_store/frames.parquet")

    # Single lookup
    frame = store.get("L21_V001_keyframe_000001")

    # Batch lookup (order preserved)
    frames = store.get_many(
        ["L21_V001_keyframe_000001", "L21_V002_keyframe_000002"]
    )

    # Temporal neighbours
    neighbors = store.get_neighbors(
        "L21_V001_keyframe_000001", window_ms=5_000
    )

"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hcmai.corpus.models import Frame


class _FrameArtifact(BaseModel):
    """Validate only the canonical fields exposed by the runtime corpus."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    frame_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    image_path: str = Field(min_length=1)
    thumbnail_path: str | None = None


class FrameStore:
    """In-memory index over canonical frame metadata.

    Loads ``frames.parquet`` once at construction time and builds three
    internal indexes for fast lookup:

    * ``_records_by_id`` – ``dict[frame_id, Frame]`` for O(1)
        point lookups.
    * ``_records_by_video`` – ``dict[video_id, tuple[Frame]]``
        sorted by ``(timestamp_ms, frame_idx, frame_id)`` for temporal
        neighbour queries.
    * ``_records`` – ``tuple[Frame]`` in the original Parquet
        row order, used for deterministic whole-corpus iteration.

    Attributes:
        metadata_path: Resolved path to the ``frames.parquet`` file
            that was loaded at construction time.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        """Load ``frames.parquet`` and build all internal indexes.

        Args:
            metadata_path: Path to the canonical ``frames.parquet`` file
            produced by ``prepare_frames`` or ``prepare_frame_store``.
            Accepts a string or any
                path-like object.

        Raises:
            FileNotFoundError: If ``metadata_path`` does not exist
                (raised by ``pd.read_parquet`` internally).
            ValueError: If the Parquet file contains duplicate
                ``frame_id`` values.
            pydantic.ValidationError: If a row has invalid runtime frame fields.
        """

        self.metadata_path = Path(metadata_path)
        table = pd.read_parquet(self.metadata_path)
        rows = cast(
            list[dict[str, object]],
            table.to_dict(orient="records"),
        )
        self._records = tuple(
            self._record_from_row(row)
            for row in rows
        )
        self._records_by_id = {
            record.frame_id: record for record in self._records
        }

        if len(self._records_by_id) != len(self._records):
            raise ValueError(
                f"Duplicate frame_id values in {self.metadata_path}"
            )

        records_by_video: defaultdict[str, list[Frame]] = defaultdict(
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
        self._submission_pairs = {
            (record.video_id, record.frame_idx) for record in self._records
        }

    @classmethod
    def load(cls, metadata_path: str | Path) -> FrameStore:
        """Load canonical frame metadata from Parquet."""

        return cls(metadata_path)

    def iter_frames(self) -> Iterator[Frame]:
        """Iterate canonical records in deterministic Parquet order."""

        return iter(self._records)

    def __len__(self) -> int:
        """Return the number of canonical records in constant time."""

        return len(self._records)

    def contains_submission(self, video_id: str, frame_idx: int) -> bool:
        """Return whether an official submission pair exists."""

        return (video_id, frame_idx) in self._submission_pairs

    @staticmethod
    def _record_from_row(row: dict[str, object]) -> Frame:
        """Validate one Parquet row and materialize its runtime frame view.

        Extracts only the six runtime fields, converts pandas ``NA`` / ``NaN``
        for ``thumbnail_path`` to ``None``, and validates the artifact boundary
        before constructing the frozen runtime dataclass.

        Args:
            row: Plain ``dict`` produced by
                ``DataFrame.to_dict(orient='records')``.
                Keys may include additional columns that are ignored.

        Returns:
            Six-field runtime ``Frame`` preserving canonical identity and paths.

        Raises:
            pydantic.ValidationError: If any required field is missing or
                has an incompatible type.
        """

        values = {
            name: row[name]
            for name in _FrameArtifact.model_fields
            if name in row
        }
        value = values.get("thumbnail_path")
        if (
            value is None
            or value is pd.NA
            or isinstance(value, float)
            and math.isnan(value)
        ):
            values["thumbnail_path"] = None
        artifact_record = _FrameArtifact.model_validate(values)
        return Frame(
            frame_id=artifact_record.frame_id,
            video_id=artifact_record.video_id,
            frame_idx=artifact_record.frame_idx,
            timestamp_ms=artifact_record.timestamp_ms,
            image_path=artifact_record.image_path,
            thumbnail_path=artifact_record.thumbnail_path,
        )

    def get(self, frame_id: str) -> Frame:
        """Return the runtime ``Frame`` for a given ``frame_id``.

        Args:
            frame_id: Globally unique internal key stored in the canonical
                Parquet file. Its text is never parsed for official IDs.

        Returns:
            The runtime ``Frame`` associated with ``frame_id``.

        Raises:
            KeyError: If ``frame_id`` is not present in the loaded
                metadata, with a descriptive message that includes the
                source file path.
        """

        try:
            return self._records_by_id[frame_id]
        except KeyError:
            raise KeyError(
                f"Unknown frame_id {frame_id!r} in {self.metadata_path}"
            ) from None

    def get_many(self, frame_ids: Sequence[str]) -> list[Frame]:
        """Return runtime ``Frame`` objects for a sequence of frame IDs.

        Preserves the order of ``frame_ids`` and allows duplicate IDs
        (the same record is returned for each occurrence).

        Args:
            frame_ids: Ordered sequence of ``frame_id`` strings to look
                up.  May contain duplicates.

        Returns:
            List of runtime ``Frame`` objects in the same order as
            ``frame_ids``.

        Raises:
            KeyError: If any ``frame_id`` in the sequence is not present
                in the loaded metadata.
        """

        return [self.get(frame_id) for frame_id in frame_ids]

    def get_by_video(self, video_id: str) -> tuple[Frame, ...]:
        """Return a video's canonical frames in deterministic temporal order.

        The returned tuple is the immutable object built at store startup and
        sorted by ``(timestamp_ms, frame_idx, frame_id)``. Unknown videos return
        an empty tuple, allowing timeline adapters to skip unavailable videos
        without fabricating frame identity.
        """

        return self._records_by_video.get(video_id, ())

    def get_neighbors(
        self,
        frame_id: str,
        *,
        window_ms: int,
        include_self: bool = False,
    ) -> list[Frame]:
        """Return same-video frames within a symmetric temporal window.

        Finds all frames in the same video whose ``timestamp_ms`` falls
        within ``[target.timestamp_ms - window_ms,
        target.timestamp_ms + window_ms]`` (inclusive on both ends).
        The target frame itself is excluded by default.

        Args:
            frame_id: ``frame_id`` of the reference frame.
            window_ms: Half-width of the temporal window in
                milliseconds.  Must be greater than or equal to zero.
            include_self: When ``True``, the reference frame is included
                in the result.  Defaults to ``False``.

        Returns:
            List of runtime ``Frame`` objects for neighbour frames, sorted
            by ``(timestamp_ms, frame_idx, frame_id)`` as stored in the
            per-video index.

        Raises:
            KeyError: If ``frame_id`` is not present in the metadata.
            ValueError: If ``window_ms`` is negative.
        """

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
