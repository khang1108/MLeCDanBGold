"""Caption artifact contract owned by offline caption enrichment."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus


class CaptionEvidence(ContractModel):
    """One frame-aligned caption with model and frame-store provenance."""

    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    text: str | None = None
    frame_store_id: NonEmptyString | None = None
    artifact_version: NonEmptyString
    model_name: NonEmptyString
    model_revision: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure_details(self) -> Self:
        """Require diagnostics when caption generation failed."""

        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed evidence requires error_code and error_message")
        return self


def usable_completed_text(row: CaptionEvidence) -> str | None:
    """Return usable completed caption text without inventing empty evidence."""

    if row.status is not ProcessingStatus.COMPLETED:
        return None
    return row.text if row.text is not None and row.text.strip() else None


__all__ = ["CaptionEvidence", "usable_completed_text"]
