"""Run YOLOE and materialize canonical object enrichment.

This module owns the offline object-detection pipeline: it resolves canonical
frames, publishes raw YOLOE JSON for resumability, and commits the canonical
``frames.parquet``/``detections.parquet``/``manifest.json`` bundle. It does
not own downstream object lookup or FrameContext assembly.
"""

from __future__ import annotations

import json
import logging
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from offline.artifact_readers import FrameAssetError, OfflineFrameAssetResolver
from offline.enrichment.models import ProcessingStatus
from offline.enrichment.objects.models import ObjectDetection, ObjectEvidence
from offline.ingestion.models import FrameArtifact

logger = logging.getLogger(__name__)


def _normalize_lineage(value: str | None, name: str) -> str | None:
    """NFC-normalize and validate an optional artifact lineage value."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class ObjectDetectionConfig:
    """Reproducibility and summary policy for one YOLOE enrichment run."""

    model: str = "yoloe-26l-seg-pf.pt"
    vocab_path: str | None = None
    min_confidence: float = 0.20
    top_k: int = 30
    batch_size: int = 32
    device: str | None = None
    artifact_version: str = "object-yoloe-v1"
    summary_min_confidence: float = 0.25
    max_summary_labels: int = 20

    def __post_init__(self) -> None:
        """Validate detector limits and the deterministic summary policy."""

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("device must be a non-empty string or null")
        if self.vocab_path is not None and (
            not isinstance(self.vocab_path, str) or not self.vocab_path.strip()
        ):
            raise ValueError("vocab_path must be a non-empty string or null")
        if not isinstance(self.artifact_version, str) or not self.artifact_version.strip():
            raise ValueError("artifact_version must not be empty")
        for name, value in (
            ("top_k", self.top_k),
            ("batch_size", self.batch_size),
            ("max_summary_labels", self.max_summary_labels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("min_confidence", self.min_confidence),
            ("summary_min_confidence", self.summary_min_confidence),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(
            self,
            "vocab_path",
            self.vocab_path.strip() if self.vocab_path is not None else None,
        )
        object.__setattr__(self, "min_confidence", float(self.min_confidence))
        object.__setattr__(
            self,
            "device",
            self.device.strip() if self.device is not None else None,
        )
        object.__setattr__(
            self,
            "artifact_version",
            unicodedata.normalize("NFC", self.artifact_version.strip()),
        )
        object.__setattr__(
            self,
            "summary_min_confidence",
            float(self.summary_min_confidence),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return stable configuration fields for stage identity and manifests."""

        return asdict(self)


def _normalized_label(value: object) -> str:
    """Return a stable label while retaining repeated detections."""

    if not isinstance(value, str):
        raise TypeError("detection label must be a string")
    collapsed = " ".join(unicodedata.normalize("NFC", value).split())
    label = unicodedata.normalize("NFC", collapsed.casefold())
    if not label:
        raise ValueError("detection label must not be empty")
    return label


def _finite_unit_number(value: object, name: str) -> float:
    """Validate one confidence or normalized box coordinate."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _parse_payload(payload: object) -> list[ObjectDetection]:
    """Parse the raw YOLOE/BTC parallel-array JSON shape strictly."""

    if not isinstance(payload, dict):
        raise TypeError("object JSON must contain an object")
    required = (
        "detection_class_entities",
        "detection_scores",
        "detection_boxes",
    )
    values: list[list[Any]] = []
    for name in required:
        value = payload.get(name)
        if not isinstance(value, list):
            raise TypeError(f"{name} must be an array")
        values.append(value)
    labels, scores, boxes = values
    if len({len(labels), len(scores), len(boxes)}) != 1:
        raise ValueError("object detection arrays must have identical length")

    detections: list[ObjectDetection] = []
    for index, (label_value, score_value, box_value) in enumerate(
        zip(labels, scores, boxes, strict=True)
    ):
        if not isinstance(box_value, list) or len(box_value) != 4:
            raise ValueError(f"detection_boxes[{index}] must contain four values")
        ymin, xmin, ymax, xmax = [
            _finite_unit_number(value, f"detection_boxes[{index}]") for value in box_value
        ]
        if ymin > ymax or xmin > xmax:
            raise ValueError(f"detection_boxes[{index}] minimum exceeds maximum")
        detections.append(
            ObjectDetection(
                label=_normalized_label(label_value),
                confidence=_finite_unit_number(score_value, f"detection_scores[{index}]"),
                x_min=xmin,
                y_min=ymin,
                x_max=xmax,
                y_max=ymax,
        ))
    return detections


def _derived_summary(
    detections: list[ObjectDetection], config: ObjectDetectionConfig
) -> tuple[dict[str, int], str | None]:
    """Build thresholded counts without discarding raw detection multiplicity."""

    retained = [
        detection
        for detection in detections
        if detection.confidence >= config.summary_min_confidence
    ]
    counts = Counter(detection.label for detection in retained)
    maximums: dict[str, float] = defaultdict(float)
    for detection in retained:
        maximums[detection.label] = max(maximums[detection.label], detection.confidence)
    labels = sorted(
        counts,
        key=lambda label: (-counts[label], -maximums[label], label),
    )
    selected = labels[: config.max_summary_labels]
    summary = "; ".join(f"{label} x{counts[label]}" for label in selected) or None
    return dict(counts), summary


def normalized_boxes(pixel_boxes: list[list[float]], width: int, height: int) -> list[list[float]]:
    """Convert pixel ``x1,y1,x2,y2`` to BTC ``ymin,xmin,ymax,xmax``."""

    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")

    def unit(value: float) -> float:
        return min(1.0, max(0.0, value))

    return [[
        unit(y_min / height),
        unit(x_min / width),
        unit(y_max / height),
        unit(x_max / width),
        ]
        for x_min, y_min, x_max, y_max in pixel_boxes
    ]


def load_vocab(path: str | Path) -> list[str]:
    """Read one prompt class per line, order-preserving and deduplicated."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required detector vocabulary not found: {source}")
    names: list[str] = []
    seen: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        name = " ".join(unicodedata.normalize("NFC", line).split()).casefold()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError(f"detector vocabulary is empty: {source}")
    return names


def _failure_evidence(
    frame: FrameArtifact,
    config: ObjectDetectionConfig,
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


def _object_path(frame: FrameArtifact, raw_output_root: Path) -> Path:
    """Map one canonical frame to its resumable raw JSON output."""

    stem = Path(frame.image_path).stem
    if not stem:
        raise ValueError("canonical image_path must have a filename stem")
    return raw_output_root / frame.video_id / f"{stem}.json"


def _frame_from_row(row: dict[str, object]) -> FrameArtifact:
    """Validate one streamed canonical frame row without retaining the store."""

    values = {name: row[name] for name in FrameArtifact.model_fields if name in row}
    return FrameArtifact.model_validate(values)


def _frame_batches(path: Path, batch_size: int = 512):
    """Yield validated canonical frames in bounded-memory batches."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield [_frame_from_row(row) for row in batch.to_pylist()]


def materialize_object_artifacts(
    frames_path: str | Path,
    raw_output_root: str | Path,
    output_dir: str | Path,
    config: ObjectDetectionConfig,
    *,
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    """Build canonical object artifacts from raw YOLOE JSON outputs."""

    from offline.enrichment.object_artifacts import write_object_artifacts_streaming

    source = Path(frames_path)
    if not source.is_file():
        raise FileNotFoundError(f"required canonical frames not found: {source}")
    normalized_frame_store_id = _normalize_lineage(frame_store_id, "frame_store_id")
    raw_root = Path(raw_output_root)
    output = Path(output_dir)
    completed = failed = frame_count = detection_count = 0

    def batches():
        nonlocal completed, failed, frame_count, detection_count
        seen_frames: set[str] = set()
        for frames in _frame_batches(source):
            # Each yield must carry only its own batch; the writer rejects a
            # frame_id it has already seen.
            evidence_rows: list[ObjectEvidence] = []
            detection_rows: list[dict[str, Any]] = []
            for frame in frames:
                if frame.frame_id in seen_frames:
                    raise ValueError("object frame rows contain duplicate frame_id values")
                seen_frames.add(frame.frame_id)
                try:
                    object_path = _object_path(frame, raw_root)
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
                        detection_rows.append({
                            "frame_id": evidence.frame_id,
                            "video_id": evidence.video_id,
                            "frame_idx": evidence.frame_idx,
                            "timestamp_ms": evidence.timestamp_ms,
                            "detection_index": detection_index,
                            **detection.model_dump(mode="json"),
                        })
                    completed += 1
                except Exception as error:
                    evidence = _failure_evidence(
                        frame,
                        config,
                        error,
                        frame_store_id=normalized_frame_store_id,
                    )
                    failed += 1
                evidence_rows.append(evidence)
            frame_count += len(evidence_rows)
            detection_count += len(detection_rows)
            yield evidence_rows, detection_rows

    manifest: dict[str, Any] = {
        "artifact_version": config.artifact_version,
        "source": "yoloe",
        "frame_store_id": normalized_frame_store_id,
        "raw_output_root": str(raw_root.resolve()),
        **config.as_dict(),
        "frame_count": 0,
        "completed_frames": 0,
        "failed_frames": 0,
        "detection_count": 0,
        "files": ["frames.parquet", "detections.parquet"],
    }

    def manifest_batches():
        yield from batches()
        manifest.update(
            frame_count=frame_count,
            completed_frames=completed,
            failed_frames=failed,
            detection_count=detection_count,
        )

    write_object_artifacts_streaming(output, manifest_batches(), manifest)
    return manifest


def pending_frames(
    frames_path: Path, raw_output_root: Path, limit: int | None
) -> list[tuple[str, str]]:
    """Return canonical frames whose raw YOLOE JSON is not published."""

    import pyarrow.parquet as pq

    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    table = pq.read_table(frames_path, columns=["video_id", "image_path"])
    pending: list[tuple[str, str]] = []
    for row in table.to_pylist():
        video_id = str(row["video_id"])
        image_path = str(row["image_path"])
        target = raw_output_root / video_id / f"{Path(image_path).stem}.json"
        if target.is_file():
            continue
        pending.append((video_id, image_path))
        if limit is not None and len(pending) >= limit:
            break
    return pending


def publish_raw_json(target: Path, payload: dict[str, list[Any]]) -> None:
    """Publish one raw detector result through a same-directory rename."""

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(".json.staged")
    staged.write_text(json.dumps(payload), encoding="utf-8")
    staged.replace(target)


def run_yoloe(
    frames_path: str | Path,
    output_dir: str | Path,
    config: ObjectDetectionConfig,
    *,
    dataset_root: str | Path = "data",
    raw_output_root: str | Path | None = None,
    frame_store_id: str | None = None,
    limit: int | None = None,
    model: Any | None = None,
    resolver: OfflineFrameAssetResolver | None = None,
) -> dict[str, Any]:
    """Detect pending frames, then commit one complete canonical artifact bundle."""

    source = Path(frames_path)
    if not source.is_file():
        raise FileNotFoundError(f"required canonical frames not found: {source}")
    output = Path(output_dir)
    raw_root = Path(raw_output_root) if raw_output_root is not None else output / "raw"
    pending = pending_frames(source, raw_root, limit)
    inference_completed = inference_skipped = 0

    if pending:
        detector: Any = model
        if detector is None:
            yolo_module = import_module("ultralytics")
            detector = yolo_module.YOLOE(config.model)
        if config.vocab_path is not None:
            names = load_vocab(config.vocab_path)
            detector.set_classes(names, detector.get_text_pe(names))
            logger.info(
                "Prompted YOLOE with %d classes from %s",
                len(names),
                config.vocab_path,
            )
        active_resolver = resolver or OfflineFrameAssetResolver(dataset_root)
        batch_starts = range(0, len(pending), config.batch_size)

        batch: list[tuple[str, Path]] = []
        for start in tqdm(batch_starts, desc="YOLOE object detection", unit="batch"):

            for video_id, image_path in pending[start : start + config.batch_size]:
                try:
                    batch.append((video_id, active_resolver.resolve_value(image_path)))
                except FrameAssetError as error:
                    inference_skipped += 1
                    logger.warning("Skipping unavailable frame: %s", error)
            if not batch:
                continue

            results = detector.predict(
                [str(image) for _, image in batch],
                conf=config.min_confidence,
                device=config.device,
                verbose=False,
            )
            if len(results) != len(batch):
                raise RuntimeError("YOLOE returned a different result count than the input batch")
            for (video_id, image), result in zip(batch, results, strict=True):
                height, width = result.orig_shape
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    payload: dict[str, list[Any]] = {
                        "detection_class_entities": [],
                        "detection_scores": [],
                        "detection_boxes": [],
                    }
                else:
                    order = boxes.conf.argsort(descending=True)[: config.top_k]
                    payload = {
                        "detection_class_entities": [
                            str(result.names[int(index)]) for index in boxes.cls[order].tolist()
                        ],
                        "detection_scores": [
                            min(1.0, max(0.0, float(score))) for score in boxes.conf[order].tolist()
                        ],
                        "detection_boxes": normalized_boxes(
                            boxes.xyxy[order].tolist(), width, height
                    ),}
                publish_raw_json(raw_root / video_id / f"{image.stem}.json", payload)
                inference_completed += 1

    manifest = materialize_object_artifacts(
        source,
        raw_root,
        output,
        config,
        frame_store_id=frame_store_id,
    )
    manifest.update(
        inference_completed_frames=inference_completed,
        inference_skipped_frames=inference_skipped,
    )
    logger.info(
        "YOLOE object enrichment complete: inference_completed=%d "
        "inference_skipped=%d artifact_completed=%d artifact_failed=%d output=%s",
        inference_completed,
        inference_skipped,
        manifest["completed_frames"],
        manifest["failed_frames"],
        output,
    )
    return manifest


__all__ = [
    "ObjectDetectionConfig",
    "load_vocab",
    "materialize_object_artifacts",
    "normalized_boxes",
    "pending_frames",
    "publish_raw_json",
    "run_yoloe",
]
