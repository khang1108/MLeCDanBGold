"""Validate and atomically publish canonical object evidence artifacts.

The object detector owns inference and raw model output. This module owns only
the stable Parquet/manifest bundle consumed by ``ObjectStore`` and
``FrameContext``. The frame table preserves canonical frame order, while the
detection table preserves every detection and its per-frame index.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hcmai.common.schemas import ObjectEvidence
from hcmai.common.utils.io import atomic_write, read_json, write_json, write_parquet
from hcmai.data.enrichment.bundle import publish_staged_bundle


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
    """Reject incomplete frame coverage and inconsistent detection identity."""

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
        _validate_artifact_tables(staged_frames, staged_detections, canonical_order)
        if read_json(staged[2]) != manifest:
            raise ValueError("staged object manifest failed round-trip validation")

        publish_staged_bundle(staged, published)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def write_object_artifacts_streaming(
    output_dir: Path,
    batches: Iterable[tuple[list[ObjectEvidence], list[dict[str, Any]]]],
    manifest: dict[str, Any],
) -> None:
    """Atomically write object artifacts while keeping one batch in memory."""

    frame_schema = pa.schema(
        [
            ("frame_id", pa.string()),
            ("video_id", pa.string()),
            ("frame_idx", pa.int64()),
            ("timestamp_ms", pa.int64()),
            ("counts_json", pa.string()),
            ("summary", pa.string()),
            ("detection_count", pa.int64()),
            ("frame_store_id", pa.string()),
            ("artifact_version", pa.string()),
            ("status", pa.string()),
            ("error_code", pa.string()),
            ("error_message", pa.string()),
        ]
    )
    detection_schema = pa.schema(
        [
            ("frame_id", pa.string()),
            ("video_id", pa.string()),
            ("frame_idx", pa.int64()),
            ("timestamp_ms", pa.int64()),
            ("detection_index", pa.int64()),
            ("label", pa.string()),
            ("confidence", pa.float64()),
            ("x_min", pa.float64()),
            ("y_min", pa.float64()),
            ("x_max", pa.float64()),
            ("y_max", pa.float64()),
        ]
    )

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
    frame_writer: pq.ParquetWriter | None = None
    detection_writer: pq.ParquetWriter | None = None
    seen_frames: set[str] = set()
    frame_count = detection_count = 0
    try:
        for evidence_rows, detection_rows in batches:
            frame_values = [frame_artifact_row(row) for row in evidence_rows]
            batch_ids = [str(row["frame_id"]) for row in frame_values]
            if len(batch_ids) != len(set(batch_ids)) or seen_frames.intersection(
                batch_ids
            ):
                raise ValueError("object frame rows contain duplicate frame_id values")
            seen_frames.update(batch_ids)

            expected_by_frame = {
                row["frame_id"]: int(row["detection_count"])
                for row in frame_values
            }
            actual_by_frame: dict[str, int] = {}
            for row in detection_rows:
                frame_id = row["frame_id"]
                if frame_id not in expected_by_frame:
                    raise ValueError(
                        "object detection references an unknown batch frame"
                    )
                actual_by_frame[frame_id] = actual_by_frame.get(frame_id, 0) + 1
            for frame_id, expected in expected_by_frame.items():
                if expected != actual_by_frame.get(frame_id, 0):
                    raise ValueError(
                        f"detection_count does not match detections for {frame_id}"
                    )

            if frame_values:
                table = pa.Table.from_pylist(frame_values, schema=frame_schema)
                if frame_writer is None:
                    frame_writer = pq.ParquetWriter(staged[0], frame_schema)
                frame_writer.write_table(table)

            detection_table = pa.Table.from_pylist(
                detection_rows, schema=detection_schema
            )
            if detection_writer is None:
                detection_writer = pq.ParquetWriter(staged[1], detection_schema)
            if detection_table.num_rows:
                detection_writer.write_table(detection_table)
            frame_count += len(frame_values)
            detection_count += len(detection_rows)

        if frame_writer is None or detection_writer is None or frame_count == 0:
            raise ValueError("canonical frame store must contain at least one frame")
        frame_writer.close()
        frame_writer = None
        detection_writer.close()
        detection_writer = None
        if pq.ParquetFile(staged[0]).metadata.num_rows != frame_count:
            raise ValueError("staged object frame row count mismatch")
        if pq.ParquetFile(staged[1]).metadata.num_rows != detection_count:
            raise ValueError("staged object detection row count mismatch")
        write_json(manifest, staged[2])
        publish_staged_bundle(staged, published)
    finally:
        if frame_writer is not None:
            frame_writer.close()
        if detection_writer is not None:
            detection_writer.close()
        for path in staged:
            path.unlink(missing_ok=True)


__all__ = [
    "DETECTION_COLUMNS",
    "FRAME_COLUMNS",
    "frame_artifact_row",
    "write_object_artifacts",
    "write_object_artifacts_streaming",
]
