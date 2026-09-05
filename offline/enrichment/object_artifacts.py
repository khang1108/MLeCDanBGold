"""Validate and atomically publish canonical object evidence artifacts.

The object detector owns inference and raw model output. This module owns only
the stable Parquet/manifest bundle consumed by ``ObjectStore`` and
``FrameContext``. The frame table preserves canonical frame order, while the
detection table preserves every detection and its per-frame index.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from offline.enrichment.objects.models import ObjectEvidence
from hcmai.common.utils.io import write_json
from offline.enrichment.bundle import publish_staged_bundle


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
    "frame_artifact_row",
    "write_object_artifacts_streaming",
]
