"""Lightweight stores used to materialize the public keyframe catalog.

These stores expose video-level organizer metadata and frame-level object
counts. They intentionally do not load raw object detections: those remain in
``ObjectStore`` for consumers that need boxes and confidence values.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hcmai.corpus.models import VideoMetadata


@dataclass(frozen=True)
class ObjectCountsRecord:
    """One object-count projection retaining canonical frame alignment."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int
    counts: dict[str, int]
    status: str


def _non_blank_text(value: object) -> str | None:
    """Return a stripped string, treating absent or blank metadata as missing."""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _object_counts(value: object, artifact_path: Path) -> dict[str, int]:
    """Decode validated non-negative object counts from one flattened row."""

    if not isinstance(value, str):
        raise ValueError(f"counts_json must be a string in {artifact_path}")
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"counts_json must contain JSON in {artifact_path}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"counts_json must contain an object in {artifact_path}")

    counts: dict[str, int] = {}
    for label, count in raw.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"object count label is invalid in {artifact_path}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"object count is invalid in {artifact_path}")
        counts[label] = count
    return counts


class ObjectCountsStore:
    """Index lightweight object label counts without materializing raw boxes."""

    def __init__(self, artifact_path: str | Path) -> None:
        """Load the object frame artifact and index each canonical frame ID."""

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise FileNotFoundError(
                f"Object counts artifact does not exist: {self.artifact_path}"
            )
        table = pd.read_parquet(self.artifact_path).astype(object)
        required = {
            "frame_id",
            "video_id",
            "frame_idx",
            "timestamp_ms",
            "counts_json",
            "status",
        }
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(
                f"{self.artifact_path} is missing columns: {', '.join(missing)}"
            )

        self._records_by_frame_id: dict[str, ObjectCountsRecord] = {}
        frame_store_ids: set[str] = set()
        for row in table.where(table.notna(), None).to_dict(orient="records"):
            frame_id = _non_blank_text(row["frame_id"])
            video_id = _non_blank_text(row["video_id"])
            if frame_id is None or video_id is None:
                raise ValueError(
                    f"Object counts identity is missing in {self.artifact_path}"
                )
            frame_idx = row["frame_idx"]
            timestamp_ms = row["timestamp_ms"]
            if (
                isinstance(frame_idx, bool)
                or not isinstance(frame_idx, int)
                or frame_idx < 0
                or isinstance(timestamp_ms, bool)
                or not isinstance(timestamp_ms, int)
                or timestamp_ms < 0
            ):
                raise ValueError(
                    f"Object counts coordinate is invalid in {self.artifact_path}"
                )
            status = row["status"]
            if status not in {"pending", "processing", "completed", "failed"}:
                raise ValueError(
                    f"Object counts status is invalid in {self.artifact_path}"
                )
            if frame_id in self._records_by_frame_id:
                raise ValueError(
                    f"Duplicate frame_id {frame_id!r} in {self.artifact_path}"
                )
            frame_store_id = _non_blank_text(row.get("frame_store_id"))
            if frame_store_id is not None:
                frame_store_ids.add(frame_store_id)
            self._records_by_frame_id[frame_id] = ObjectCountsRecord(
                frame_id=frame_id,
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                counts=_object_counts(row["counts_json"], self.artifact_path),
                status=status,
            )
        if len(frame_store_ids) > 1:
            raise ValueError(
                f"Object counts use multiple frame_store_id values in "
                f"{self.artifact_path}"
            )
        self.frame_store_id = next(iter(frame_store_ids), None)

    def get_counts(self, frame_id: str) -> dict[str, int] | None:
        """Return completed counts, preserving empty results and missing status."""

        record = self._records_by_frame_id.get(frame_id)
        if record is None or record.status != "completed":
            return None
        return dict(record.counts)

    def iter_records(self) -> Iterator[ObjectCountsRecord]:
        """Iterate stored object-count records in artifact order."""

        return iter(self._records_by_frame_id.values())


class VideoMetadataStore:
    """Index title and watch URL from one organizer media-info directory."""

    def __init__(self, metadata_root: str | Path) -> None:
        """Load every ``{video_id}.json`` organizer record once at startup."""

        self.metadata_root = Path(metadata_root)
        if not self.metadata_root.is_dir():
            raise NotADirectoryError(
                f"Video metadata directory does not exist: {self.metadata_root}"
            )
        self._by_video_id: dict[str, VideoMetadata] = {}
        for path in sorted(self.metadata_root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid video metadata JSON: {path}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"Video metadata must be an object: {path}")
            video_id = path.stem
            self._by_video_id[video_id] = VideoMetadata(
                video_id=video_id,
                title=_non_blank_text(raw.get("title")),
                video_url=_non_blank_text(raw.get("watch_url")),
            )

    def get(self, video_id: str) -> VideoMetadata | None:
        """Return metadata for a video, or ``None`` when no JSON record exists."""

        return self._by_video_id.get(video_id)
