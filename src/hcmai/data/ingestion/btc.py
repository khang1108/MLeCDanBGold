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


logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "btc-keyframe-ingestion-v1"
_SOURCE = "btc_provided_keyframes"
_REQUIRED_COLUMNS = frozenset(
    {
        "frame_id",
        "video_id",
        "frame_idx",
        "keyframe_order",
        "timestamp_ms",
        "image_path",
        "width",
        "height",
    }
)


@dataclass(frozen=True)
class BTCIngestionConfig:
    """Paths and lineage used to materialize one canonical BTC frame store."""

    btc_root: Path
    data_root: Path
    output_root: Path
    frame_store_id: str

    def __post_init__(self) -> None:
        if not self.frame_store_id.strip():
            raise ValueError("frame_store_id must not be empty")


def _compute_fps_per_video(frames: pd.DataFrame) -> dict[str, float]:
    """Estimate per-video FPS while preserving BTC frame indices unchanged."""

    fps_by_video: dict[str, float] = {}
    for video_id, group in frames.groupby("video_id", sort=False):
        valid = group[group["timestamp_ms"] > 0]
        if len(valid) < 2:
            fps_by_video[str(video_id)] = 30.0
            continue

        estimates = valid["frame_idx"] / (valid["timestamp_ms"] / 1000.0)
        fps = float(np.median(estimates))
        for standard_fps in (24.0, 25.0, 30.0):
            if abs(fps - standard_fps) < 1.5:
                fps = standard_fps
                break
        fps_by_video[str(video_id)] = fps

    return fps_by_video


def _validate_source_columns(frames: pd.DataFrame, source_path: Path) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frames.columns))
    if missing:
        raise ValueError(
            f"BTC metadata {source_path} is missing required columns: "
            f"{', '.join(missing)}"
        )


def _resolve_image_path(data_root: Path, image_path: object) -> str:
    if image_path is None or pd.isna(image_path):
        raise ValueError("image_path must not be null")
    normalized = str(image_path).strip()
    if not normalized:
        raise ValueError("image_path must not be blank")
    return str((data_root / normalized).resolve())


def _validated_record(values: dict[str, object]) -> FrameRecord:
    """Validate one normalized canonical row through the shared contract."""

    for name in ("thumbnail_path", "shot_id", "event_id", "pts", "time_base"):
        value = values.get(name)
        if value is None or value is pd.NA:
            values[name] = None
        elif isinstance(value, float) and np.isnan(value):
            values[name] = None
    return FrameRecord.model_validate(values)


def _validate_unique_identity(records: list[FrameRecord]) -> None:
    seen_frame_ids: set[str] = set()
    seen_submission_coordinates: set[tuple[str, int]] = set()
    for record in records:
        if record.frame_id in seen_frame_ids:
            raise ValueError(f"Duplicate frame_id: {record.frame_id}")
        seen_frame_ids.add(record.frame_id)

        coordinate = (record.video_id, record.frame_idx)
        if coordinate in seen_submission_coordinates:
            raise ValueError(
                "Duplicate submission coordinate: "
                f"video_id={record.video_id}, frame_idx={record.frame_idx}"
            )
        seen_submission_coordinates.add(coordinate)


def _validate_canonical_table(frames: pd.DataFrame) -> list[FrameRecord]:
    records = [
        _validated_record(dict(row))
        for row in cast(
            list[dict[str, Any]], frames.to_dict(orient="records")
        )
    ]
    _validate_unique_identity(records)
    return records


def _build_canonical_rows(
    source_frames: pd.DataFrame,
    *,
    data_root: Path,
    fps_by_video: dict[str, float],
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
                    "fps": fps_by_video.get(str(video_id), 30.0),
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
    _validate_unique_identity(canonical_records)
    return pd.DataFrame(
        [record.model_dump() for record in canonical_records],
        columns=list(FrameRecord.model_fields),
    )


def _warn_about_missing_images(frames: pd.DataFrame) -> None:
    missing_paths = [
        image_path
        for image_path in frames["image_path"]
        if not Path(str(image_path)).is_file()
    ]
    if not missing_paths:
        logger.info("All BTC keyframe images exist")
        return

    logger.warning("%d BTC keyframe images were not found", len(missing_paths))
    for image_path in missing_paths[:5]:
        logger.warning("Missing BTC keyframe image: %s", image_path)


def import_btc_frame_store(config: BTCIngestionConfig) -> Path:
    """Materialize BTC metadata as a canonical, lineage-stamped frame store.

    The importer never invokes preprocessing. Source frame identifiers, frame
    indices, keyframe order, and timestamps are preserved; only their pandas
    scalar types are normalized for stable Parquet serialization.
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
    source_frames = source_frames.sort_values(
        ["video_id", "timestamp_ms"], kind="stable"
    ).reset_index(drop=True)

    fps_by_video = _compute_fps_per_video(source_frames)
    canonical = _build_canonical_rows(
        source_frames,
        data_root=config.data_root,
        fps_by_video=fps_by_video,
    )
    _warn_about_missing_images(canonical)

    video_count = len(set(canonical["video_id"].astype(str).tolist()))
    manifest = {
        "pipeline_version": _PIPELINE_VERSION,
        "source": _SOURCE,
        "frame_store_id": config.frame_store_id,
        "btc_root": str(config.btc_root.resolve()),
        "video_count": video_count,
        "frame_count": int(len(canonical)),
        "limited_run": False,
        "resume_enabled": False,
        "fps_map_sample": dict(list(fps_by_video.items())[:5]),
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

        staged_frames.replace(output_path)
        # The manifest is the final commit marker for the published bundle.
        staged_manifest.replace(manifest_path)
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
