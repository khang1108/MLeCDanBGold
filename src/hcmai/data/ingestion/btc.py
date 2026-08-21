"""Import BTC-provided keyframes into the canonical frame store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from hcmai.common.schemas import FrameRecord
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)
from hcmai.data.ingestion.keyframe_map import join_btc_mapping, load_btc_keyframe_map


logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "btc-keyframe-ingestion-v1"
_SOURCE = "btc_provided_keyframes"
_REQUIRED_COLUMNS = frozenset(
    {
        "frame_id",
        "video_id",
        "keyframe_order",
        "image_path",
        "width",
        "height",
    }
)


@dataclass(frozen=True)
class BTCIngestionConfig:
    """Paths and lineage used to materialize one canonical BTC frame store."""

    btc_root: Path
    mapping_root: Path
    data_root: Path
    output_root: Path
    frame_store_id: str

    def __post_init__(self) -> None:
        if not self.frame_store_id.strip():
            raise ValueError("frame_store_id must not be empty")


def _validate_source_columns(frames: pd.DataFrame, source_path: Path) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frames.columns))
    if missing:
        raise ValueError(
            f"BTC metadata {source_path} is missing required columns: "
            f"{', '.join(missing)}"
        )


def _resolve_image_path(data_root: Path, image_path: object) -> str:
    """Return a portable path relative to the canonical ``data`` root."""

    if (
        image_path is None
        or image_path is pd.NA
        or isinstance(image_path, float)
        and np.isnan(image_path)
    ):
        raise ValueError("image_path must not be null")
    if not isinstance(image_path, str):
        raise ValueError("image_path must be a string")
    normalized = str(image_path).strip()
    if not normalized:
        raise ValueError("image_path must not be blank")
    root = data_root.expanduser().resolve()
    source = Path(normalized).expanduser()
    resolved = source.resolve() if source.is_absolute() else (root / source).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("image_path must remain under data_root")
    return resolved.relative_to(root).as_posix()


def _validated_record(values: dict[str, object]) -> FrameRecord:
    """Validate one normalized canonical row through the shared contract."""

    for name in ("thumbnail_path", "shot_id", "event_id", "pts", "time_base"):
        value = values.get(name)
        if value is None or value is pd.NA:
            values[name] = None
        elif isinstance(value, float) and np.isnan(value):
            values[name] = None
    return FrameRecord.model_validate(values)


def _validate_unique_frame_ids(records: list[FrameRecord]) -> None:
    """Reject ambiguous internal identities without rewriting BTC coordinates.

    Multiple BTC keyframes may legitimately map to the same competition-facing
    ``(video_id, frame_idx)``. Their distinct ``frame_id`` values remain the
    authoritative identities for joins and enrichment.
    """

    seen_frame_ids: set[str] = set()
    for record in records:
        if record.frame_id in seen_frame_ids:
            raise ValueError(f"Duplicate frame_id: {record.frame_id}")
        seen_frame_ids.add(record.frame_id)


def _validate_canonical_table(frames: pd.DataFrame) -> list[FrameRecord]:
    records = [
        _validated_record(dict(row))
        for row in cast(
            list[dict[str, Any]], frames.to_dict(orient="records")
        )
    ]
    _validate_unique_frame_ids(records)
    return records


def _submission_coordinate_diagnostics(frames: pd.DataFrame) -> dict[str, int]:
    """Summarize non-injective competition coordinates for observability."""

    multiplicities = frames.groupby(
        ["video_id", "frame_idx"], sort=False, dropna=False
    ).size()
    collisions = multiplicities[multiplicities > 1]
    collision_counts = cast(list[int], collisions.astype("int64").tolist())
    all_counts = cast(list[int], multiplicities.astype("int64").tolist())
    return {
        "duplicate_submission_coordinate_groups": len(collision_counts),
        "duplicate_submission_coordinate_rows": sum(collision_counts),
        "maximum_submission_coordinate_multiplicity": max(
            all_counts, default=0
        ),
    }


def _build_canonical_rows(
    source_frames: pd.DataFrame,
    *,
    data_root: Path,
) -> pd.DataFrame:
    records = cast(list[dict[str, Any]], source_frames.to_dict(orient="records"))
    canonical_records: list[FrameRecord] = []
    for row in records:
        video_id = row["video_id"]
        canonical_records.append(
            _validated_record(
                {
                    "frame_id": row["frame_id"],
                    "video_id": video_id,
                    # BTC frame_idx is the authoritative competition coordinate.
                    "frame_idx": row["frame_idx"],
                    "keyframe_order": row["keyframe_order"],
                    "timestamp_ms": row["timestamp_ms"],
                    # The BTC map carries exact per-video media FPS.
                    "fps": row["fps"],
                    "image_path": _resolve_image_path(
                        data_root, row["image_path"]
                    ),
                    "thumbnail_path": None,
                    "width": row["width"],
                    "height": row["height"],
                    "shot_id": None,
                    "event_id": None,
                    "is_anchor": True,
                    # BTC metadata does not establish PTS/time-base semantics.
                    "pts": None,
                    "time_base": None,
                    "motion_score": 0.0,
                    "shot_score": 0.0,
                    "event_score": 0.0,
                    "selection_reasons": ("btc_keyframe",),
                }
            )
        )
    _validate_unique_frame_ids(canonical_records)
    return pd.DataFrame(
        [record.model_dump() for record in canonical_records],
        columns=list(FrameRecord.model_fields),
    )


def _warn_about_missing_images(frames: pd.DataFrame, data_root: Path) -> None:
    """Report missing images after resolving portable paths under data_root."""

    root = data_root.expanduser().resolve()
    missing_paths = [
        image_path
        for image_path in frames["image_path"]
        if not (root / str(image_path)).is_file()
    ]
    if not missing_paths:
        logger.info("All BTC keyframe images exist")
        return

    logger.warning("%d BTC keyframe images were not found", len(missing_paths))
    for image_path in missing_paths[:5]:
        logger.warning("Missing BTC keyframe image: %s", image_path)


def _publish_staged_bundle(
    *,
    staged_frames: Path,
    staged_manifest: Path,
    output_path: Path,
    manifest_path: Path,
) -> None:
    """Publish both staged files or restore the complete previous bundle."""

    frames_backup = output_path.parent / ".frames.parquet.backup"
    manifest_backup = manifest_path.parent / ".manifest.json.backup"
    for backup in (frames_backup, manifest_backup):
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite stale backup: {backup}")

    frames_publish_attempted = False
    manifest_publish_attempted = False
    cleanup_backups = False
    try:
        if output_path.exists():
            output_path.replace(frames_backup)
        if manifest_path.exists():
            manifest_path.replace(manifest_backup)

        frames_publish_attempted = True
        staged_frames.replace(output_path)
        # The manifest is the final commit marker for the published bundle.
        manifest_publish_attempted = True
        staged_manifest.replace(manifest_path)
        cleanup_backups = True
    except Exception:
        if manifest_publish_attempted:
            manifest_path.unlink(missing_ok=True)
        if frames_publish_attempted:
            output_path.unlink(missing_ok=True)
        if frames_backup.exists():
            frames_backup.replace(output_path)
        if manifest_backup.exists():
            # Restore the previous manifest last as its commit marker.
            manifest_backup.replace(manifest_path)
        cleanup_backups = True
        raise
    finally:
        if cleanup_backups:
            frames_backup.unlink(missing_ok=True)
            manifest_backup.unlink(missing_ok=True)


def import_btc_frame_store(config: BTCIngestionConfig) -> Path:
    """Materialize BTC metadata as a canonical, lineage-stamped frame store.

    The importer never invokes preprocessing. Internal source frame identities
    are preserved, while BTC map_keyframes supplies authoritative competition
    coordinates, timestamps, and FPS.
    """

    source_path = config.btc_root / "metadata" / "frames.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(f"BTC metadata not found: {source_path}")

    logger.info("Reading BTC metadata from %s", source_path)
    source_frames = pd.read_parquet(source_path)
    _validate_source_columns(source_frames, source_path)
    if source_frames.empty:
        raise ValueError(
            f"BTC metadata {source_path} must contain at least one frame"
        )
    mapping = load_btc_keyframe_map(config.mapping_root)
    source_frames = join_btc_mapping(source_frames, mapping).sort_values(
        ["video_id", "keyframe_order"], kind="stable"
    ).reset_index(drop=True)

    canonical = _build_canonical_rows(
        source_frames,
        data_root=config.data_root,
    )
    _warn_about_missing_images(canonical, config.data_root)

    video_count = len(set(canonical["video_id"].astype(str).tolist()))
    mapping_video_count = len(set(mapping["video_id"].astype(str).tolist()))
    manifest = {
        "pipeline_version": _PIPELINE_VERSION,
        "source": _SOURCE,
        "frame_store_id": config.frame_store_id,
        "btc_root": str(config.btc_root.resolve()),
        "mapping_root": str(config.mapping_root.resolve()),
        "video_count": video_count,
        "frame_count": int(len(canonical)),
        "mapping_video_count": mapping_video_count,
        "mapping_row_count": int(len(mapping)),
        "limited_run": False,
        "resume_enabled": False,
        "fps_map_sample": dict(
            canonical.groupby("video_id", sort=True)["fps"].first().head(5).items()
        ),
        **_submission_coordinate_diagnostics(canonical),
    }

    output_path = config.output_root / "frames.parquet"
    manifest_path = config.output_root / "manifest.json"
    staged_frames = config.output_root / ".frames.parquet.staged"
    staged_manifest = config.output_root / ".manifest.json.staged"
    try:
        atomic_write(
            staged_frames,
            lambda staging: write_parquet(canonical, staging, index=False),
        )
        atomic_write(
            staged_manifest,
            lambda staging: write_json(manifest, staging),
        )

        _validate_canonical_table(pd.read_parquet(staged_frames))
        if read_json(staged_manifest) != manifest:
            raise ValueError("Staged BTC manifest failed round-trip validation")

        _publish_staged_bundle(
            staged_frames=staged_frames,
            staged_manifest=staged_manifest,
            output_path=output_path,
            manifest_path=manifest_path,
        )
    finally:
        staged_frames.unlink(missing_ok=True)
        staged_manifest.unlink(missing_ok=True)

    logger.info(
        "Wrote %d frames across %d videos to %s",
        len(canonical),
        video_count,
        output_path,
    )
    return output_path


__all__ = ["BTCIngestionConfig", "import_btc_frame_store"]
