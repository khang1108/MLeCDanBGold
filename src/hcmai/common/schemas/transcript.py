from __future__ import annotations

from pydantic import Field, model_validator

from hcmai.common.schemas.base import ContractModel, NonEmptyString


class TranscriptSegment(ContractModel):
    """Canonical transcript metadata for one spoken segment."""

    segment_id: NonEmptyString
    video_id: NonEmptyString
    segment_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    text: NonEmptyString
    language: NonEmptyString
    speaker_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> TranscriptSegment:
        """Require every transcript segment to have a positive duration."""

        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self
