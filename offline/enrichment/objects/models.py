"""Object-detection artifact contracts and config owned by offline enrichment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Self
import unicodedata

from pydantic import Field, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus


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
        from dataclasses import asdict
        return asdict(self)


class ObjectDetection(ContractModel):
    """One normalized BTC-provided detection without parent identity fields."""

    label: NonEmptyString
    confidence: float = Field(ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_box(self) -> Self:
        """Reject inverted normalized detection boxes."""
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("object maximum coordinates must not precede minimums")
        return self


class ObjectEvidence(ContractModel):
    """All BTC detections and a thresholded summary for one frame."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    detections: list[ObjectDetection] = Field(default_factory=list)
    counts: dict[NonEmptyString, int] = Field(default_factory=dict)
    summary: str | None = None
    detection_count: int = Field(default=0, ge=0)
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Validate failure details and retained raw label multiplicity."""
        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        raw_counts = Counter(detection.label for detection in self.detections)
        if self.detection_count != len(self.detections):
            raise ValueError("detection_count must equal the number of detections")
        if any(
            count < 0 or count > raw_counts.get(label, 0)
            for label, count in self.counts.items()
        ):
            raise ValueError("counts must not exceed raw detection multiplicity")
        return self


__all__ = ["ObjectDetection", "ObjectDetectionConfig", "ObjectEvidence"]
