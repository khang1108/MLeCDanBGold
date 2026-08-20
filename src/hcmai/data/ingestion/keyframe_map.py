"""Load and apply BTC organizer keyframe coordinates.

This module owns the strict boundary between source frame metadata and the
organizer-provided ``map_keyframes`` CSV files. It preserves internal
``frame_id`` values while making BTC frame coordinates, timestamps, and FPS
authoritative. It does not write canonical artifacts or perform image
inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


_MAPPING_COLUMNS = ["n", "pts_time", "fps", "frame_idx"]
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def _finite_numeric_column(
    table: pd.DataFrame,
    column: str,
    path: Path,
    *,
    description: str = "finite numeric",
) -> pd.Series:
    """Return one numeric mapping column after rejecting malformed values."""

    raw_values = cast(pd.Series, table[column])
    values = cast(pd.Series, pd.to_numeric(raw_values, errors="coerce")).astype(float)
    numeric_values = np.asarray(values, dtype=float)
    if bool(values.isna().any()) or not bool(np.isfinite(numeric_values).all()):
        raise ValueError(
            f"BTC mapping {column} must contain {description} values: {path}"
        )
    return values


def _finite_integral_column(
    table: pd.DataFrame, column: str, path: Path
) -> pd.Series:
    """Return a finite integral mapping column without truncating its values."""

    values = _finite_numeric_column(
        table,
        column,
        path,
        description="finite integral",
    )
    numeric_values = np.asarray(values, dtype=float)
    if not bool(np.equal(numeric_values, np.floor(numeric_values)).all()):
        raise ValueError(
            f"BTC mapping {column} must contain finite integral values: {path}"
        )
    return values.astype("int64")


def load_btc_keyframe_map(mapping_root: Path) -> pd.DataFrame:
    """Load strict BTC maps keyed by video ID and organizer keyframe order.

    Each CSV must have the organizer schema, contiguous one-based ``n``, one
    positive FPS value, and non-negative temporal/submission coordinates.
    """

    rows: list[pd.DataFrame] = []
    for path in sorted(mapping_root.glob("*.csv")):
        table = pd.read_csv(path)
        if list(table.columns) != _MAPPING_COLUMNS:
            raise ValueError(f"Unexpected BTC mapping schema: {path}")

        n_values = _finite_integral_column(table, "n", path)
        frame_idx_values = _finite_integral_column(table, "frame_idx", path)
        pts_time_values = _finite_numeric_column(table, "pts_time", path)
        fps_values = _finite_numeric_column(table, "fps", path)

        expected = list(range(1, len(table) + 1))
        if n_values.tolist() != expected:
            raise ValueError(f"BTC mapping n must be contiguous 1..N: {path}")
        if bool((pts_time_values < 0).any()) or bool((frame_idx_values < 0).any()):
            raise ValueError(f"BTC mapping coordinates must be non-negative: {path}")

        distinct_fps_values = fps_values.unique()
        if len(distinct_fps_values) != 1 or float(distinct_fps_values[0]) <= 0:
            raise ValueError(
                f"BTC mapping fps must be one positive value per video: {path}"
            )

        rows.append(
            table.assign(
                n=n_values,
                pts_time=pts_time_values,
                fps=fps_values,
                frame_idx=frame_idx_values,
                video_id=path.stem,
                keyframe_order=n_values,
            )
        )

    if not rows:
        raise ValueError(f"No BTC mapping CSV files found under {mapping_root}")

    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["video_id", "keyframe_order"]).any():
        raise ValueError("BTC mapping contains duplicate video/keyframe order")
    return result


def join_btc_mapping(source_frames: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Apply organizer coordinates without changing source ``frame_id`` values.

    Legacy metadata coordinates are intentionally discarded before the join:
    BTC mapping is the authoritative source for ``frame_idx``, ``fps``, and
    ``timestamp_ms``. Repeated ``(video_id, frame_idx)`` pairs remain valid.
    """

    source = source_frames.drop(
        columns=["frame_idx", "timestamp_ms", "fps", "pts_time"],
        errors="ignore",
    )
    joined = source.merge(
        mapping[["video_id", "keyframe_order", "pts_time", "fps", "frame_idx"]],
        on=["video_id", "keyframe_order"],
        how="left",
        validate="one_to_one",
    )
    pts_time = cast(pd.Series, joined["pts_time"])
    if bool(pts_time.isna().any()):
        raise ValueError("Canonical source contains frames missing from BTC mapping")

    joined["frame_idx"] = joined["frame_idx"].astype("int64")
    joined["fps"] = joined["fps"].astype(float)
    joined["timestamp_ms"] = (
        joined["pts_time"].astype(float) * 1000.0
    ).round().astype("int64")
    return joined


def project_keyframe_paths(frames: pd.DataFrame, keyframes_root: Path) -> pd.DataFrame:
    """Return a copy with image paths projected onto locally staged keyframes.

    Files are matched by sorted filename to each video's ascending
    ``keyframe_order``. Only ``image_path`` is rewritten, so canonical
    identities and organizer coordinates remain machine-independent.
    """

    projected = frames.copy()
    for video_id, video_frames in projected.groupby("video_id", sort=False):
        image_directory = keyframes_root / str(video_id)
        images: list[Path] = []
        if image_directory.is_dir():
            images = sorted(
                path
                for path in image_directory.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            )
        ordered_indices = video_frames.sort_values(
            "keyframe_order", kind="stable"
        ).index.tolist()
        if len(images) != len(ordered_indices):
            raise ValueError(
                "Staged keyframe count does not match canonical frame count "
                f"for {video_id}: {len(images)} != {len(ordered_indices)}"
            )
        for index, image_path in zip(ordered_indices, images, strict=True):
            projected.at[index, "image_path"] = str(image_path)
    return projected


__all__ = [
    "join_btc_mapping",
    "load_btc_keyframe_map",
    "project_keyframe_paths",
]
