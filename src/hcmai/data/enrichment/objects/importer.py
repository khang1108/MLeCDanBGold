"""Strictly normalize BTC TensorFlow/OpenImages object JSON artifacts.

Each canonical frame produces one evidence row. Source failures are contained
to that frame; the importer performs no model inference or schema guessing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any
import unicodedata

from hcmai.common.schemas import (
    FrameRecord,
    ObjectDetection,
    ObjectEvidence,
    ProcessingStatus,
)
from hcmai.data.stores.frame import FrameStore

from .artifacts import write_object_artifacts
from .config import ObjectConfig, normalize_lineage


def _normalized_label(value: object) -> str:
    """Return the canonical label without losing repeated detections."""

    if not isinstance(value, str):
        raise TypeError("detection label must be a string")
    collapsed = " ".join(unicodedata.normalize("NFC", value).split())
    label = unicodedata.normalize("NFC", collapsed.casefold())
    if not label:
        raise ValueError("detection label must not be empty")
    return label


def _finite_unit_number(value: object, name: str) -> float:
    """Validate one normalized score or coordinate."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _parse_payload(payload: object) -> list[ObjectDetection]:
    """Parse only the documented BTC TensorFlow parallel-array shape."""

    if not isinstance(payload, dict):
        raise TypeError("BTC object JSON must contain an object")
    required = (
        "detection_class_entities",
        "detection_scores",
        "detection_boxes",
    )
    values = []
    for name in required:
        value = payload.get(name)
        if not isinstance(value, list):
            raise TypeError(f"{name} must be an array")
        values.append(value)
    labels, scores, boxes = values
    if len({len(labels), len(scores), len(boxes)}) != 1:
        raise ValueError("BTC object detection arrays must have identical length")

    detections: list[ObjectDetection] = []
    for index, (label_value, score_value, box_value) in enumerate(
        zip(labels, scores, boxes)
    ):
        if not isinstance(box_value, list) or len(box_value) != 4:
            raise ValueError(f"detection_boxes[{index}] must contain four values")
        ymin, xmin, ymax, xmax = [
            _finite_unit_number(value, f"detection_boxes[{index}]")
            for value in box_value
        ]
        if ymin > ymax or xmin > xmax:
            raise ValueError(f"detection_boxes[{index}] minimum exceeds maximum")
        detections.append(
            ObjectDetection(
                label=_normalized_label(label_value),
                confidence=_finite_unit_number(
                    score_value, f"detection_scores[{index}]"
                ),
                x_min=xmin,
                y_min=ymin,
                x_max=xmax,
                y_max=ymax,
            )
        )
    return detections


def _derived_summary(
    detections: list[ObjectDetection], config: ObjectConfig
) -> tuple[dict[str, int], str | None]:
    """Build thresholded counts and a deterministic non-spatial summary."""

    retained = [
        detection
        for detection in detections
        if detection.confidence >= config.summary_min_confidence
    ]
    counts = Counter(detection.label for detection in retained)
    maximums: dict[str, float] = defaultdict(float)
    for detection in retained:
        maximums[detection.label] = max(
            maximums[detection.label], detection.confidence
        )
    labels = sorted(counts, key=lambda label: (-counts[label], -maximums[label], label))
    selected = labels[: config.max_summary_labels]
    summary = "; ".join(f"{label} x{counts[label]}" for label in selected) or None
    return dict(counts), summary


def _failure_evidence(
    frame: FrameRecord,
    config: ObjectConfig,
    error: Exception,
    *,
    frame_store_id: str | None,
) -> ObjectEvidence:
    """Create one failed evidence row with bounded diagnostics."""

    message = " ".join(str(error).split())[:300] or type(error).__name__
    return ObjectEvidence(
        frame_id=frame.frame_id,
        video_id=frame.video_id,
        frame_idx=frame.frame_idx,
        timestamp_ms=frame.timestamp_ms,
        frame_store_id=frame_store_id,
        artifact_version=config.artifact_version,
        status=ProcessingStatus.FAILED,
        error_code=type(error).__name__,
        error_message=message,
    )


def _object_path(frame: FrameRecord, objects_root: Path) -> Path:
    """Map BTC keyframe identity to its documented sibling JSON name."""

    stem = Path(frame.image_path).stem
    if not stem:
        raise ValueError("canonical image_path must have a filename stem")
    return objects_root / frame.video_id / f"{stem}.json"


def _load_canonical_frames(path: Path) -> list[FrameRecord]:
    """Load canonical rows keyed by their unique internal ``frame_id``."""

    store = FrameStore.load(path)
    frames = list(store.iter_frames())
    if not frames:
        raise ValueError(
            f"canonical frame store {path} must contain at least one frame"
        )

    return frames


def import_objects(
    frames_path: str | Path,
    objects_root: str | Path,
    output_dir: str | Path,
    config: ObjectConfig,
    *,
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    """Import BTC detections while containing missing or malformed frame JSON."""

    source = Path(frames_path)
    if not source.is_file():
        raise FileNotFoundError(f"required canonical frames not found: {source}")
    frames = _load_canonical_frames(source)
    canonical_order = [frame.frame_id for frame in frames]
    normalized_frame_store_id = normalize_lineage(
        frame_store_id, "frame_store_id"
    )

    root = Path(objects_root)
    evidence_rows: list[ObjectEvidence] = []
    detection_rows: list[dict[str, Any]] = []
    completed = failed = 0
    for frame in frames:
        try:
            object_path = _object_path(frame, root)
            with object_path.open("r", encoding="utf-8") as file:
                detections = _parse_payload(json.load(file))
            counts, summary = _derived_summary(detections, config)
            evidence = ObjectEvidence(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                timestamp_ms=frame.timestamp_ms,
                detections=detections,
                counts=counts,
                summary=summary,
                detection_count=len(detections),
                frame_store_id=normalized_frame_store_id,
                artifact_version=config.artifact_version,
            )
            for detection_index, detection in enumerate(detections):
                detection_rows.append(
                    {
                        "frame_id": evidence.frame_id,
                        "video_id": evidence.video_id,
                        "frame_idx": evidence.frame_idx,
                        "timestamp_ms": evidence.timestamp_ms,
                        "detection_index": detection_index,
                        **detection.model_dump(mode="json"),
                    }
                )
            completed += 1
        except Exception as error:
            evidence = _failure_evidence(
                frame, config, error, frame_store_id=normalized_frame_store_id
            )
            failed += 1
        evidence_rows.append(evidence)

    output = Path(output_dir)
    manifest = {
        "artifact_version": config.artifact_version,
        "source": "btc_provided_objects",
        "frame_store_id": normalized_frame_store_id,
        "objects_root": str(root.resolve()),
        "frame_count": len(frames),
        "completed_frames": completed,
        "failed_frames": failed,
        "detection_count": len(detection_rows),
        "summary_min_confidence": config.summary_min_confidence,
        "max_summary_labels": config.max_summary_labels,
        "files": ["frames.parquet", "detections.parquet"],
    }
    write_object_artifacts(
        output, canonical_order, evidence_rows, detection_rows, manifest
    )
    return manifest


__all__ = ["import_objects"]
