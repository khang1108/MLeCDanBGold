"""OCR artifact contracts owned by offline OCR enrichment."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus


class OCRRegion(ContractModel):
    """One raw OCR region aligned to its canonical parent frame."""

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
    def validate_box(self) -> Self:
        """Reject inverted normalized region boxes."""

        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("OCR region maximum coordinates must not precede minimums")
        return self


class OCREvidence(ContractModel):
    """Frame-level OCR evidence retaining raw and normalized text."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
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

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require diagnostics when OCR generation failed."""

        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


def usable_completed_text(row: OCREvidence) -> str | None:
    """Return usable completed normalized OCR text."""

    if row.status is not ProcessingStatus.COMPLETED:
        return None
    value = row.normalized_text
    return value if value is not None and value.strip() else None


__all__ = ["OCREvidence", "OCRRegion", "usable_completed_text"]
