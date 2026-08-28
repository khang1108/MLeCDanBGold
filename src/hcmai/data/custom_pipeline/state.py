"""Persist local archive, batch, and video resume state.

State is stored as one atomic JSON file per archive, deterministic batch, and
video under ``runs/<version>/state/{archives,batches,videos}/``. This module
owns only the higher-level local pipeline lifecycle; it never reads or edits
the separate native C++ per-video extraction state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.common.utils.logging import get_logger
from hcmai.data.custom_pipeline.config import ArchiveWorkWindow
from hcmai.data.custom_pipeline.contracts import RunIdentity

logger = get_logger(__name__)

_MAX_FAILURE_HISTORY = 5
_MAX_VIDEOS_PER_BATCH = 8
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class ArchiveStage(str, Enum):
    """Lifecycle of one downloaded/extracted organizer ZIP."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    EXTRACTED = "extracted"
    PROCESSING = "processing"
    COMPLETE = "complete"
    CLEANED = "cleaned"


class BatchStage(str, Enum):
    """Lifecycle of one canonical group of at most eight videos."""

    PLANNED = "planned"
    EXTRACTED = "extracted"
    ARTIFACTS_COMPLETE = "artifacts_complete"
    INDEXES_COMPLETE = "indexes_complete"
    COMMITTED = "committed"
    EPHEMERAL_CLEANED = "ephemeral_cleaned"


class VideoStage(str, Enum):
    """Per-video progress through the local specialist/embedding stages."""

    PENDING = "pending"
    SOURCE_READY = "source_ready"
    EXTRACTED = "extracted"
    CAPTIONED = "captioned"
    OCR_COMPLETE = "ocr_complete"
    OBJECTS_COMPLETE = "objects_complete"
    CONTEXT_COMPLETE = "context_complete"
    EMBEDDINGS_COMPLETE = "embeddings_complete"
    LOCAL_COMPLETE = "local_complete"


_ARCHIVE_ORDER = list(ArchiveStage)
_BATCH_ORDER = list(BatchStage)
_VIDEO_ORDER = list(VideoStage)


def _require_safe_id(value: str, *, field_name: str) -> str:
    """Reject blank or path-unsafe identifiers before they become filenames."""

    if not value or not _SAFE_ID_PATTERN.match(value):
        raise ValueError(f"{field_name} must be a non-blank safe identifier: {value!r}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_forward_transition(order: list[Enum], current: Enum, new: Enum, *, kind: str) -> None:
    """Allow only an idempotent replay or exactly one forward step.

    Skipping a step (e.g. ``extracted -> committed``) or moving backward would
    hide a real missing stage or silently discard validated progress.
    """

    current_index = order.index(current)
    new_index = order.index(new)
    if new_index == current_index:
        return
    if new_index != current_index + 1:
        raise ValueError(
            f"{kind} stage transition {current.value!r} -> {new.value!r} is "
            "not an allowed single forward step"
        )


def compute_batch_id(archive_id: str, batch_index: int) -> str:
    """Return the deterministic ID for the ``batch_index``-th group in an archive."""

    _require_safe_id(archive_id, field_name="archive_id")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    return f"{archive_id}-batch{batch_index:03d}"


def _append_bounded_failure(failures: list[dict[str, Any]], diagnostic: dict[str, Any]) -> list[dict[str, Any]]:
    """Append one failure entry while keeping only the most recent history."""

    return [*failures, {**diagnostic, "recorded_at": _now()}][-_MAX_FAILURE_HISTORY:]


@dataclass
class ArchiveRecord:
    """Resume state for one organizer ZIP at its fixed plan position."""

    archive_id: str
    position: int
    stage: ArchiveStage = ArchiveStage.PENDING
    failures: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_id": self.archive_id,
            "position": self.position,
            "stage": self.stage.value,
            "failures": self.failures,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchiveRecord":
        return cls(
            archive_id=data["archive_id"],
            position=data["position"],
            stage=ArchiveStage(data["stage"]),
            failures=list(data.get("failures", [])),
            updated_at=data.get("updated_at", _now()),
        )


@dataclass
class BatchRecord:
    """Resume state for one canonical group of at most eight videos."""

    batch_id: str
    archive_id: str
    video_ids: list[str]
    stage: BatchStage = BatchStage.PLANNED
    failures: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "archive_id": self.archive_id,
            "video_ids": self.video_ids,
            "stage": self.stage.value,
            "failures": self.failures,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchRecord":
        return cls(
            batch_id=data["batch_id"],
            archive_id=data["archive_id"],
            video_ids=list(data["video_ids"]),
            stage=BatchStage(data["stage"]),
            failures=list(data.get("failures", [])),
            updated_at=data.get("updated_at", _now()),
        )


@dataclass
class VideoRecord:
    """Resume state for one video inside its owning batch."""

    video_id: str
    batch_id: str
    stage: VideoStage = VideoStage.PENDING
    failures: list[dict[str, Any]] = field(default_factory=list)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "batch_id": self.batch_id,
            "stage": self.stage.value,
            "failures": self.failures,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoRecord":
        return cls(
            video_id=data["video_id"],
            batch_id=data["batch_id"],
            stage=VideoStage(data["stage"]),
            failures=list(data.get("failures", [])),
            updated_at=data.get("updated_at", _now()),
        )


class PipelineStateStore:
    """Atomic local archive/batch/video resume state for one run root."""

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)
        self.state_root = self.run_root / "state"
        self._run_path = self.state_root / "run.json"
        self._archives_dir = self.state_root / "archives"
        self._batches_dir = self.state_root / "batches"
        self._videos_dir = self.state_root / "videos"

    # -- run identity ------------------------------------------------------

    def create_or_resume_run(
        self,
        identity: RunIdentity,
        work_window: ArchiveWorkWindow,
    ) -> dict[str, Any]:
        """Create a new run or resume an existing one with a matching identity.

        Raises:
            ValueError: If a prior run exists with a different identity, or the
                requested work window's offset skips an archive that is not
                yet ``cleaned``.
        """

        if self._run_path.exists():
            existing = read_json(self._run_path)
            existing_identity = RunIdentity(**existing["identity"])
            if existing_identity != identity:
                raise ValueError(
                    "run identity changed; cannot resume prior local state "
                    f"under {self.run_root}"
                )
            windows: list[dict[str, Any]] = list(existing["work_windows"])
        else:
            windows = []

        self._require_no_gap_before_offset(work_window.offset)
        windows.append(
            {
                "offset": work_window.offset,
                "limit": work_window.limit,
                "accepted_at": _now(),
            }
        )
        record = {"identity": identity.model_dump(), "work_windows": windows}
        atomic_write(self._run_path, lambda p: write_json(record, p))
        logger.info(
            "run state accepted work window offset=%d limit=%s (%d total attempts)",
            work_window.offset,
            work_window.limit,
            len(windows),
        )
        return record

    def _require_no_gap_before_offset(self, offset: int) -> None:
        """Ensure every archive position before ``offset`` is already cleaned."""

        for position in range(offset):
            record = self._read_archive_by_position(position)
            if record is None or record.stage != ArchiveStage.CLEANED:
                raise ValueError(
                    f"cannot start at offset {offset}: archive at position "
                    f"{position} is not cleaned"
                )

    def _read_archive_by_position(self, position: int) -> ArchiveRecord | None:
        if not self._archives_dir.exists():
            return None
        for path in sorted(self._archives_dir.glob("*.json")):
            data = read_json(path)
            if data.get("position") == position:
                return ArchiveRecord.from_dict(data)
        return None

    # -- archive lifecycle ---------------------------------------------------

    def _archive_path(self, archive_id: str) -> Path:
        safe_id = _require_safe_id(archive_id, field_name="archive_id")
        return self._archives_dir / f"{safe_id}.json"

    def get_archive(self, archive_id: str) -> ArchiveRecord | None:
        path = self._archive_path(archive_id)
        if not path.exists():
            return None
        return ArchiveRecord.from_dict(read_json(path))

    def ensure_archive(self, archive_id: str, position: int) -> ArchiveRecord:
        """Return the existing archive record or create it as ``pending``."""

        existing = self.get_archive(archive_id)
        if existing is not None:
            return existing
        record = ArchiveRecord(archive_id=archive_id, position=position)
        self._write_archive(record)
        logger.info("archive %s registered at position %d", archive_id, position)
        return record

    def advance_archive(self, archive_id: str, stage: ArchiveStage) -> ArchiveRecord:
        record = self.get_archive(archive_id)
        if record is None:
            raise ValueError(f"unknown archive_id: {archive_id}")
        _check_forward_transition(_ARCHIVE_ORDER, record.stage, stage, kind="archive")
        record.stage = stage
        record.updated_at = _now()
        self._write_archive(record)
        logger.info("archive %s -> %s", archive_id, stage.value)
        return record

    def record_archive_failure(self, archive_id: str, diagnostic: dict[str, Any]) -> ArchiveRecord:
        record = self.get_archive(archive_id)
        if record is None:
            raise ValueError(f"unknown archive_id: {archive_id}")
        record.failures = _append_bounded_failure(record.failures, diagnostic)
        record.updated_at = _now()
        self._write_archive(record)
        logger.warning("archive %s recorded failure: %s", archive_id, diagnostic)
        return record

    def _write_archive(self, record: ArchiveRecord) -> None:
        atomic_write(self._archive_path(record.archive_id), lambda p: write_json(record.to_dict(), p))

    # -- batch lifecycle -------------------------------------------------

    def _batch_path(self, batch_id: str) -> Path:
        safe_id = _require_safe_id(batch_id, field_name="batch_id")
        return self._batches_dir / f"{safe_id}.json"

    def get_batch(self, batch_id: str) -> BatchRecord | None:
        path = self._batch_path(batch_id)
        if not path.exists():
            return None
        return BatchRecord.from_dict(read_json(path))

    def ensure_batch(self, batch_id: str, archive_id: str, video_ids: list[str]) -> BatchRecord:
        """Return the existing batch record or create it as ``planned``.

        Raises:
            ValueError: If ``video_ids`` exceeds the eight-video batch ceiling.
        """

        if len(video_ids) > _MAX_VIDEOS_PER_BATCH:
            raise ValueError(
                f"batch {batch_id} has {len(video_ids)} videos, exceeding the "
                "eight-video ceiling"
            )
        existing = self.get_batch(batch_id)
        if existing is not None:
            return existing
        record = BatchRecord(batch_id=batch_id, archive_id=archive_id, video_ids=list(video_ids))
        self._write_batch(record)
        logger.info("batch %s registered with %d videos", batch_id, len(video_ids))
        return record

    def advance_batch(self, batch_id: str, stage: BatchStage) -> BatchRecord:
        """Advance one batch, keeping the furthest stage when an uncommitted batch replays."""

        record = self.get_batch(batch_id)
        if record is None:
            raise ValueError(f"unknown batch_id: {batch_id}")
        if _BATCH_ORDER.index(stage) < _BATCH_ORDER.index(record.stage):
            return record
        _check_forward_transition(_BATCH_ORDER, record.stage, stage, kind="batch")
        record.stage = stage
        record.updated_at = _now()
        self._write_batch(record)
        logger.info("batch %s -> %s", batch_id, stage.value)
        return record

    def record_batch_failure(self, batch_id: str, diagnostic: dict[str, Any]) -> BatchRecord:
        record = self.get_batch(batch_id)
        if record is None:
            raise ValueError(f"unknown batch_id: {batch_id}")
        record.failures = _append_bounded_failure(record.failures, diagnostic)
        record.updated_at = _now()
        self._write_batch(record)
        logger.warning("batch %s recorded failure: %s", batch_id, diagnostic)
        return record

    def _write_batch(self, record: BatchRecord) -> None:
        atomic_write(self._batch_path(record.batch_id), lambda p: write_json(record.to_dict(), p))

    # -- video lifecycle -------------------------------------------------

    def _video_path(self, video_id: str) -> Path:
        safe_id = _require_safe_id(video_id, field_name="video_id")
        return self._videos_dir / f"{safe_id}.json"

    def get_video(self, video_id: str) -> VideoRecord | None:
        path = self._video_path(video_id)
        if not path.exists():
            return None
        return VideoRecord.from_dict(read_json(path))

    def ensure_video(self, video_id: str, batch_id: str) -> VideoRecord:
        """Return the existing video record or create it as ``pending``."""

        existing = self.get_video(video_id)
        if existing is not None:
            return existing
        record = VideoRecord(video_id=video_id, batch_id=batch_id)
        self._write_video(record)
        return record

    def advance_video(self, video_id: str, stage: VideoStage) -> VideoRecord:
        """Advance one video, keeping the furthest stage when an uncommitted batch replays."""

        record = self.get_video(video_id)
        if record is None:
            raise ValueError(f"unknown video_id: {video_id}")
        if _VIDEO_ORDER.index(stage) < _VIDEO_ORDER.index(record.stage):
            return record
        _check_forward_transition(_VIDEO_ORDER, record.stage, stage, kind="video")
        record.stage = stage
        record.updated_at = _now()
        self._write_video(record)
        logger.info("video %s -> %s", video_id, stage.value)
        return record

    def record_video_failure(self, video_id: str, diagnostic: dict[str, Any]) -> VideoRecord:
        record = self.get_video(video_id)
        if record is None:
            raise ValueError(f"unknown video_id: {video_id}")
        record.failures = _append_bounded_failure(record.failures, diagnostic)
        record.updated_at = _now()
        self._write_video(record)
        logger.warning("video %s recorded failure: %s", video_id, diagnostic)
        return record

    def _write_video(self, record: VideoRecord) -> None:
        atomic_write(self._video_path(record.video_id), lambda p: write_json(record.to_dict(), p))

    # -- cleanup guard -----------------------------------------------------

    def require_ephemeral_cleanup_allowed(self, batch_id: str) -> BatchRecord:
        """Confirm a batch may have its ephemeral inputs removed.

        Raises:
            ValueError: If the batch is not yet ``committed``/``ephemeral_cleaned``,
                or any of its videos has not reached ``local_complete``.
        """

        record = self.get_batch(batch_id)
        if record is None:
            raise ValueError(f"unknown batch_id: {batch_id}")
        if record.stage not in (BatchStage.COMMITTED, BatchStage.EPHEMERAL_CLEANED):
            raise ValueError(
                f"batch {batch_id} is {record.stage.value!r}; ephemeral cleanup "
                "requires committed or ephemeral_cleaned"
            )
        for video_id in record.video_ids:
            video = self.get_video(video_id)
            if video is None or video.stage != VideoStage.LOCAL_COMPLETE:
                observed = video.stage.value if video is not None else "missing"
                raise ValueError(
                    f"batch {batch_id} video {video_id} is {observed!r}, not "
                    "local_complete; ephemeral cleanup is forbidden"
                )
        return record


__all__ = [
    "ArchiveRecord",
    "ArchiveStage",
    "BatchRecord",
    "BatchStage",
    "PipelineStateStore",
    "VideoRecord",
    "VideoStage",
    "compute_batch_id",
]
