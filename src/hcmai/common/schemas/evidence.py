"""Specialist evidence and deterministic frame-context contracts."""

from __future__ import annotations

from collections import Counter

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import ProcessingStatus


class _SpecialistEvidence(ContractModel):
    """Shared status validation for independently materialized evidence."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def _validate_failure_details(self) -> "_SpecialistEvidence":
        if self.status == ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


class CaptionEvidence(_SpecialistEvidence):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    text: str | None = None
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None


class OCRRegion(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    region_id: NonEmptyString
    region_order: int = Field(ge=0)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_box(self) -> "OCRRegion":
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximum coordinates must not precede minimums")
        return self


class OCREvidence(_SpecialistEvidence):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    raw_text: str | None = None
    normalized_text: str | None = None
    quality_score: float = Field(default=0.0, ge=0, le=1)
    region_count: int = Field(default=0, ge=0)
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None


class ObjectDetection(ContractModel):
    label: NonEmptyString
    confidence: float = Field(ge=0, le=1)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_box(self) -> "ObjectDetection":
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("object maximum coordinates must not precede minimums")
        return self


class ObjectEvidence(_SpecialistEvidence):
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
    def _validate_detections(self) -> "ObjectEvidence":
        raw_counts = Counter(detection.label for detection in self.detections)
        if self.detection_count != len(self.detections):
            raise ValueError("detection_count must equal the number of detections")
        if any(
            count < 0 or count > raw_counts.get(label, 0)
            for label, count in self.counts.items()
        ):
            raise ValueError("counts must not exceed raw detection multiplicity")
        return self


class FrameContext(ContractModel):
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    caption_text: str | None = None
    ocr_text: str | None = None
    object_summary: str | None = None
    context_text: str | None = None
    caption_available: bool = False
    ocr_quality: float = Field(default=0.0, ge=0, le=1)
    object_count: int = Field(default=0, ge=0)
    context_version: NonEmptyString
    caption_version: NonEmptyString
    ocr_version: NonEmptyString
    object_version: NonEmptyString
    frame_store_id: NonEmptyString | None = None


def usable_completed_text(row: CaptionEvidence | OCREvidence) -> str | None:
    """Return usable completed text without treating empty evidence as a match."""

    if row.status != ProcessingStatus.COMPLETED:
        return None
    text = row.text if isinstance(row, CaptionEvidence) else row.normalized_text
    return text if text is not None and text.strip() else None


__all__ = [
    "CaptionEvidence",
    "FrameContext",
    "ObjectDetection",
    "ObjectEvidence",
    "OCREvidence",
    "OCRRegion",
    "usable_completed_text",
]
