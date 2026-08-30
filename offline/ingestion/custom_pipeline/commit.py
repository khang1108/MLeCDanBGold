"""Atomically commit one local batch and clean only its ephemeral inputs.

A batch's durable keyframes, specialist parquet shards, embedding vectors,
and three retrieval indexes are validated and committed as one atomic
directory rename. Ephemeral cleanup (OCR scratch, native staging links,
source MP4s) is only ever permitted after that commit succeeds and every
video in the batch has reached ``local_complete``.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.common.utils.logging import get_logger
from offline.ingestion.custom_pipeline.state import BatchStage, PipelineStateStore
from hcmai.retrieval.retriever.artifacts import sha256_file
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

# Every per-video artifact a committed batch must contain. Child tables
# (ocr_regions, object_detections) may be empty but must still exist as
# files, matching the shards written by ``split_batch_artifacts_by_video``.
_REQUIRED_VIDEO_ARTIFACTS = (
    "caption.parquet",
    "ocr_frames.parquet",
    "ocr_regions.parquet",
    "object_frames.parquet",
    "object_detections.parquet",
    "context.parquet",
    "visual_vectors.npy",
    "visual_mapping.parquet",
    "context_vectors.npy",
    "context_mapping.parquet",
)
_REQUIRED_INDEX_DIRS = ("visual", "context", "asr_segments")
_MANIFEST_FILENAME = "manifest.json"
_SUCCESS_FILENAME = "_SUCCESS.json"


class BatchValidationError(RuntimeError):
    """Raised when a staged local batch fails commit validation."""


@dataclass(frozen=True)
class FileInventoryEntry:
    """One hashed file recorded in a batch's local commit inventory."""

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BatchInventory:
    """Deterministic local inventory of one staged batch's payload."""

    batch_id: str
    video_ids: tuple[str, ...]
    canonical_frame_digest: str
    files: tuple[FileInventoryEntry, ...] = field(default_factory=tuple)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_frame_digest(staging_root: Path, video_ids: Sequence[str]) -> str:
    """Hash every video's canonical ``(video_id, frame_id)`` pair.

    The ``context`` shard is used because Task 6 guarantees it has exact,
    duplicate-free frame coverage for every video in the batch.
    """

    import hashlib

    pairs: list[str] = []
    for video_id in video_ids:
        table_path = staging_root / "videos" / video_id / "context.parquet"
        if not table_path.is_file():
            raise BatchValidationError(
                f"missing canonical context shard for {video_id}: {table_path}"
            )
        table = pd.read_parquet(table_path, columns=["video_id", "frame_id"])
        pairs.extend(f"{row.video_id}:{row.frame_id}" for row in table.itertuples())

    ordered = "\n".join(sorted(pairs))
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def build_batch_inventory(
    staging_root: str | Path, batch_id: str, video_ids: Sequence[str]
) -> BatchInventory:
    """Hash every regular file under ``staging_root`` into a deterministic inventory.

    Raises:
        BatchValidationError: If the canonical ``context`` shard for any video
            is missing.
    """

    root = Path(staging_root)
    entries = [
        FileInventoryEntry(
            relative_path=str(path.relative_to(root)),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    inventory = BatchInventory(
        batch_id=batch_id,
        video_ids=tuple(video_ids),
        canonical_frame_digest=_canonical_frame_digest(root, video_ids),
        files=tuple(entries),
    )
    logger.info(
        "built batch inventory for %s: %d file(s), %d video(s)",
        batch_id,
        len(entries),
        len(video_ids),
    )
    return inventory


def _require_inventory_matches_disk(staging_root: Path, inventory: BatchInventory) -> None:
    """Re-verify every inventoried file still matches disk before commit."""

    for entry in inventory.files:
        path = staging_root / entry.relative_path
        if not path.is_file():
            raise BatchValidationError(f"inventoried file is missing on disk: {entry.relative_path}")
        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            raise BatchValidationError(
                f"inventoried size mismatch for {entry.relative_path}: "
                f"expected={entry.size_bytes} actual={actual_size}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry.sha256:
            raise BatchValidationError(f"inventoried checksum mismatch for {entry.relative_path}")


def validate_local_batch(
    batch_id: str,
    video_ids: Sequence[str],
    staging_root: str | Path,
    inventory: BatchInventory,
) -> None:
    """Validate a staged batch is complete and consistent before commit.

    Requires every specialist/vector artifact for every video, checksum-loads
    all three indexes, and confirms the inventory still matches disk.

    Raises:
        BatchValidationError: If the inventory disagrees with the requested
            batch/video IDs, any per-video artifact is missing, any index
            fails to checksum-load, or a file disagrees with its inventory.
    """

    staging_root = Path(staging_root)
    if inventory.batch_id != batch_id or tuple(inventory.video_ids) != tuple(video_ids):
        raise BatchValidationError(
            f"inventory identity ({inventory.batch_id}, {inventory.video_ids}) does not "
            f"match requested ({batch_id}, {tuple(video_ids)})"
        )

    for video_id in video_ids:
        video_dir = staging_root / "videos" / video_id
        missing = [name for name in _REQUIRED_VIDEO_ARTIFACTS if not (video_dir / name).is_file()]
        if missing:
            raise BatchValidationError(f"video {video_id} is missing artifacts: {missing}")

    try:
        DenseIndex.load(staging_root / "visual")
        DenseIndex.load(staging_root / "context")
        SegmentDenseIndex.load(staging_root / "asr_segments")
    except Exception as error:
        raise BatchValidationError(f"batch {batch_id} index checksum-load failed: {error}") from error

    _require_inventory_matches_disk(staging_root, inventory)
    logger.info("batch %s passed local commit validation", batch_id)


def commit_local_batch(
    staging_root: str | Path,
    final_batch_root: str | Path,
    inventory: BatchInventory,
) -> Path:
    """Write commit markers and atomically publish a validated staged batch.

    A conflicting, already-completed destination is accepted only when its
    recorded inventory matches exactly; any other conflict is rejected.

    Raises:
        BatchValidationError: If ``final_batch_root`` exists without a
            completed manifest, or with a manifest that disagrees.
    """

    staging_root = Path(staging_root)
    final_batch_root = Path(final_batch_root)

    manifest = {
        "batch_id": inventory.batch_id,
        "video_ids": list(inventory.video_ids),
        "canonical_frame_digest": inventory.canonical_frame_digest,
        "files": [asdict(entry) for entry in inventory.files],
    }
    # The marker files are written last, after every payload file already
    # exists, so a reader never observes a marker without its payload.
    atomic_write(staging_root / _MANIFEST_FILENAME, lambda p: write_json(manifest, p))
    atomic_write(
        staging_root / _SUCCESS_FILENAME,
        lambda p: write_json({"batch_id": inventory.batch_id, "committed_at": _now()}, p),
    )

    if final_batch_root.exists():
        existing_manifest_path = final_batch_root / _MANIFEST_FILENAME
        existing_success_path = final_batch_root / _SUCCESS_FILENAME
        if not existing_manifest_path.is_file() or not existing_success_path.is_file():
            raise BatchValidationError(
                f"conflicting incomplete batch destination: {final_batch_root}"
            )
        if read_json(existing_manifest_path) != manifest:
            raise BatchValidationError(
                f"conflicting completed batch destination with a different "
                f"inventory: {final_batch_root}"
            )
        # Already committed with an identical inventory: treat as resume.
        shutil.rmtree(staging_root)
        logger.info("batch %s already committed identically; discarded restaging", inventory.batch_id)
        return final_batch_root

    final_batch_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.replace(final_batch_root)
    logger.info("committed batch %s to %s", inventory.batch_id, final_batch_root)
    return final_batch_root


def cleanup_ephemeral_batch(
    state_store: PipelineStateStore,
    batch_id: str,
    ephemeral_paths: Sequence[str | Path],
    *,
    allowed_root: str | Path,
) -> None:
    """Remove only the given ephemeral paths after batch commit is proven safe.

    Every path must resolve inside ``allowed_root`` (e.g. the run's
    ``active/`` tree); this function never touches ``data/``, ``artifacts/``,
    or ``artifacts_legacy/``.

    Raises:
        ValueError: If the batch is not ``committed``/``ephemeral_cleaned``,
            any video is not ``local_complete`` (raised by the state store),
            or a requested path escapes ``allowed_root``.
    """

    state_store.require_ephemeral_cleanup_allowed(batch_id)
    allowed = Path(allowed_root).resolve()

    resolved_paths: list[Path] = []
    for raw_path in ephemeral_paths:
        path = Path(raw_path).resolve()
        if path == allowed or allowed not in path.parents:
            raise ValueError(
                f"refusing to clean up path outside allowed root {allowed}: {path}"
            )
        resolved_paths.append(path)

    for path in resolved_paths:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    logger.info(
        "cleaned %d ephemeral path(s) for batch %s under %s",
        len(resolved_paths),
        batch_id,
        allowed,
    )
    state_store.advance_batch(batch_id, BatchStage.EPHEMERAL_CLEANED)


__all__ = [
    "BatchInventory",
    "BatchValidationError",
    "FileInventoryEntry",
    "build_batch_inventory",
    "cleanup_ephemeral_batch",
    "commit_local_batch",
    "validate_local_batch",
]
