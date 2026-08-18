"""Contracts for canonical competition submission rows."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString


class SubmissionResult(ContractModel):
    """Official KIS competition submission code."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: NonEmptyString
    
    submission_code: NonEmptyString

    @model_validator(mode="after")
    def validate_submission_code(self) -> Self:
        """Keep the public code coupled to official frame identifiers."""
        if self.submission_code != f"{self.video_id},{self.frame_idx}":
            raise ValueError("submission_code must equal 'video_id,frame_idx'")
        return self
