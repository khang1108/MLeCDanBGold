"""Validate and atomically persist normalized BTC object artifacts.

The frame table preserves one row per canonical frame, while the detection
table preserves every valid source detection in organizer order.
"""

from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from hcmai.common.schemas import ObjectEvidence
from hcmai.common.utils.io import (
    atomic_write,
    read_json,
    write_json,
    write_parquet,
)


FRAME_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_ms",
    "counts_json",
    "summary",
    "detection_count",
    "frame_store_id",
    "artifact_version",
    "status",
    "error_code",
    "error_message",
]

DETECTION_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_idx",
    "timestamp_ms",
    "detection_index",
    "label",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
]


def frame_artifact_row(evidence: ObjectEvidence) -> dict[str, Any]:
    """Flatten one validated evidence object for stable Parquet storage."""

    return {
        "frame_id": evidence.frame_id,
        "video_id": evidence.video_id,
        "frame_idx": evidence.frame_idx,
        "timestamp_ms": evidence.timestamp_ms,
        "counts_json": json.dumps(
            dict(sorted(evidence.counts.items())),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "summary": evidence.summary,
        "detection_count": evidence.detection_count,
        "frame_store_id": evidence.frame_store_id,
        "artifact_version": evidence.artifact_version,
        "status": evidence.status.value,
        "error_code": evidence.error_code,
        "error_message": evidence.error_message,
    }


def _validate_artifact_tables(
    frame_table: pd.DataFrame,
    detection_table: pd.DataFrame,
    canonical_order: list[str],
) -> None:
    """Reject incomplete frame coverage and duplicate detection identity."""

    frame_ids = frame_table["frame_id"].astype(str).tolist()
    if frame_ids != canonical_order:
        raise ValueError("object frame rows must match canonical frame order exactly")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("object frame rows contain duplicate frame_id values")

    identity = detection_table[["frame_id", "detection_index"]]
    if identity.duplicated().any():
        raise ValueError(
            "object detections contain duplicate frame_id+detection_index identity"
        )
    if not set(detection_table["frame_id"].astype(str)).issubset(set(frame_ids)):
        raise ValueError("object detection references an unknown canonical frame")

    canonical = {
        row["frame_id"]: (
            row["video_id"],
            row["frame_idx"],
            row["timestamp_ms"],
        )
        for row in frame_table.to_dict(orient="records")
    }
    for row in detection_table.to_dict(orient="records"):
        frame_id = row["frame_id"]
        if (
            not isinstance(frame_id, str)
            or not frame_id
            or frame_id.strip() != frame_id
            or not isinstance(row["video_id"], str)
            or not row["video_id"]
            or row["video_id"].strip() != row["video_id"]
            or isinstance(row["frame_idx"], bool)
            or not isinstance(row["frame_idx"], Integral)
            or isinstance(row["timestamp_ms"], bool)
            or not isinstance(row["timestamp_ms"], Integral)
            or canonical.get(frame_id)
            != (row["video_id"], row["frame_idx"], row["timestamp_ms"])
        ):
            raise ValueError(
                "object detection canonical identity does not match its frame row"
            )

    expected = detection_table.groupby("frame_id", sort=False).size().to_dict()
    for row in frame_table.to_dict(orient="records"):
        if int(row["detection_count"]) != int(expected.get(row["frame_id"], 0)):
            raise ValueError(
                f"detection_count does not match detections for {row['frame_id']}"
            )


def _publish_staged_bundle(
    staged: tuple[Path, Path, Path],
    published: tuple[Path, Path, Path],
) -> None:
    """Publish all staged files or restore the prior complete bundle."""

    backups = tuple(
        target.with_name(f".{target.name}.backup") for target in published
    )
    for backup in backups:
        if backup.exists():
            raise RuntimeError(f"refusing to overwrite stale backup: {backup}")

    replaced: list[Path] = []
    restore_complete = False
    try:
        for target, backup in zip(published, backups):
            if target.exists():
                target.replace(backup)

        for source, target in zip(staged, published):
            replaced.append(target)
            # Manifest is deliberately last and remains the bundle commit marker.
            source.replace(target)
    except Exception:
        for target in replaced:
            target.unlink(missing_ok=True)
        # Restore data files before restoring the previous manifest marker.
        for target, backup in zip(published, backups):
            if backup.exists():
                backup.replace(target)
        restore_complete = True
        raise
    else:
        restore_complete = True
    finally:
        if restore_complete:
            for backup in backups:
                backup.unlink(missing_ok=True)


def write_object_artifacts(
    output_dir: Path,
    canonical_order: list[str],
    evidence_rows: list[ObjectEvidence],
    detection_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Validate and atomically write frame, detection, and manifest files."""

    frame_table = pd.DataFrame(
        [frame_artifact_row(row) for row in evidence_rows],
        columns=FRAME_COLUMNS,
    )
    detection_table = pd.DataFrame(detection_rows, columns=DETECTION_COLUMNS)
    _validate_artifact_tables(frame_table, detection_table, canonical_order)

    output_dir.mkdir(parents=True, exist_ok=True)
    published = (
        output_dir / "frames.parquet",
        output_dir / "detections.parquet",
        output_dir / "manifest.json",
    )
    staged = (
        output_dir / ".frames.parquet.staged",
        output_dir / ".detections.parquet.staged",
        output_dir / ".manifest.json.staged",
    )
    try:
        atomic_write(
            staged[0], lambda path: write_parquet(frame_table, path, index=False)
        )
        atomic_write(
            staged[1],
            lambda path: write_parquet(detection_table, path, index=False),
        )
        atomic_write(staged[2], lambda path: write_json(manifest, path))

        staged_frames = pd.read_parquet(staged[0])
        staged_detections = pd.read_parquet(staged[1])
        _validate_artifact_tables(
            staged_frames, staged_detections, canonical_order
        )
        if read_json(staged[2]) != manifest:
            raise ValueError("staged object manifest failed round-trip validation")

        _publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


__all__ = ["write_object_artifacts"]
