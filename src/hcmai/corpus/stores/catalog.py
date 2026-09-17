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
from typing import Any, cast

import pandas as pd

from hcmai.corpus.models import VideoMetadata


class ObjectCountsRecord:
    """One object-count projection retaining canonical frame alignment."""

    __slots__ = (
        "frame_id",
        "video_id",
        "frame_idx",
        "timestamp_ms",
        "_counts_raw",
        "status",
        "_counts_cache",
        "artifact_path",
    )

    def __init__(
        self,
        frame_id: str,
        video_id: str,
        frame_idx: int,
        timestamp_ms: int,
        counts: dict[str, int] | str,
        status: str,
        artifact_path: Path | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.timestamp_ms = timestamp_ms
        self.status = status
        self.artifact_path = artifact_path
        if isinstance(counts, dict):
            self._counts_raw = None
            self._counts_cache = counts
        else:
            self._counts_raw = counts
            self._counts_cache = None

    @property
    def counts(self) -> dict[str, int]:
        if self._counts_cache is not None:
            return self._counts_cache
        parsed = _object_counts(
            self._counts_raw, self.artifact_path or Path("<object_counts>")
        )
        self._counts_cache = parsed
        return parsed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectCountsRecord):
            return False
        return (
            self.frame_id == other.frame_id
            and self.video_id == other.video_id
            and self.frame_idx == other.frame_idx
            and self.timestamp_ms == other.timestamp_ms
            and self.status == other.status
            and self.counts == other.counts
        )

    def __repr__(self) -> str:
        return (
            f"ObjectCountsRecord(frame_id={self.frame_id!r}, "
            f"video_id={self.video_id!r}, frame_idx={self.frame_idx}, "
            f"timestamp_ms={self.timestamp_ms}, status={self.status!r})"
        )


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
        table = pd.read_parquet(self.artifact_path)
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

        fids = table["frame_id"].tolist()
        vids = table["video_id"].tolist()
        fidxs = table["frame_idx"].tolist()
        tss = table["timestamp_ms"].tolist()
        counts_jsons = table["counts_json"].tolist()
        statuses = table["status"].tolist()

        if "frame_store_id" in table.columns:
            fs_ids = table["frame_store_id"].dropna().unique().tolist()
            if len(fs_ids) > 1:
                raise ValueError(
                    f"Object counts use multiple frame_store_id values in "
                    f"{self.artifact_path}"
                )
            self.frame_store_id = fs_ids[0] if fs_ids else None
        else:
            self.frame_store_id = None

        records = [
            ObjectCountsRecord(
                frame_id=fid,
                video_id=vid,
                frame_idx=int(fidx),
                timestamp_ms=int(ts),
                counts=cj,
                status=st,
                artifact_path=self.artifact_path,
            )
            for fid, vid, fidx, ts, cj, st in zip(
                fids, vids, fidxs, tss, counts_jsons, statuses
            )
        ]
        self._records_by_frame_id = {r.frame_id: r for r in records}
        if len(self._records_by_frame_id) != len(records):
            seen = set()
            for r in records:
                if r.frame_id in seen:
                    raise ValueError(
                        f"Duplicate frame_id {r.frame_id!r} in {self.artifact_path}"
                    )
                seen.add(r.frame_id)

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
