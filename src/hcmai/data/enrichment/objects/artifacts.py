"""Validate and atomically persist normalized BTC object artifacts.

The frame table preserves one row per canonical frame, while the detection
table preserves every valid source detection in organizer order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hcmai.common.schemas import ObjectEvidence
from hcmai.common.utils.io import atomic_write, write_json, write_parquet


FRAME_COLUMNS = [
    "frame_id",
    "video_id",
    "frame_idx",
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

    expected = detection_table.groupby("frame_id", sort=False).size().to_dict()
    for row in frame_table.to_dict(orient="records"):
        if int(row["detection_count"]) != int(expected.get(row["frame_id"], 0)):
            raise ValueError(
                f"detection_count does not match detections for {row['frame_id']}"
            )


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
    atomic_write(
        output_dir / "frames.parquet",
        lambda path: write_parquet(frame_table, path, index=False),
    )
    atomic_write(
        output_dir / "detections.parquet",
        lambda path: write_parquet(detection_table, path, index=False),
    )
    # The manifest is written last and acts as the complete bundle marker.
    atomic_write(
        output_dir / "manifest.json",
        lambda path: write_json(manifest, path),
    )


__all__ = ["write_object_artifacts"]
