"""Object-detection artifact contracts owned by offline enrichment."""

from __future__ import annotations

from collections import Counter
from typing import Self

from pydantic import Field, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus


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


__all__ = ["ObjectDetection", "ObjectEvidence"]
