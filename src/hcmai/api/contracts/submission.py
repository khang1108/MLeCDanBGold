"""Public HTTP contract for canonical competition submission rows."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SubmissionResult(BaseModel):
    """Official KIS competition submission code for one canonical frame."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    frame_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    frame_idx: int = Field(ge=0)
    submission_code: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_submission_code(self) -> Self:
        """Keep the public code coupled to official frame identifiers."""

        if self.submission_code != f"{self.video_id},{self.frame_idx}":
            raise ValueError("submission_code must equal 'video_id,frame_idx'")
        return self


__all__ = ["SubmissionResult"]
