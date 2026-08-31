"""Canonical segment-native transcript artifact contract."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from offline.contracts import ContractModel, NonEmptyString
from offline.enrichment.models import ProcessingStatus


class TranscriptSegment(ContractModel):
    """Timestamped ASR artifact segment with optional model provenance."""

    segment_id: NonEmptyString
    video_id: NonEmptyString
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: NonEmptyString
    language: NonEmptyString
    speaker_id: NonEmptyString | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    model_name: NonEmptyString | None = None
    model_revision: NonEmptyString | None = None
    artifact_version: NonEmptyString = "asr-segment-v1"
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        """Require positive duration and diagnostics for failed segments."""

        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.status is ProcessingStatus.FAILED and (
            self.error_code is None or self.error_message is None
        ):
            raise ValueError("failed segments require error_code and error_message")
        return self


__all__ = ["TranscriptSegment"]
