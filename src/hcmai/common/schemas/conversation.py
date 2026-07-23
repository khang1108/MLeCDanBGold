"""Contracts for conversational known-item search."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from hcmai.common.schemas.base import ContractModel, NonEmptyString


class ConversationTurn(ContractModel):
    """One ordered message in a KISC session."""

    turn_id: NonEmptyString
    sender: Literal["user", "ai"]
    message: NonEmptyString
    created_at: int = Field(ge=0, description="Timestamp in milliseconds.")
    reply_to_turn_id: NonEmptyString | None = None


class ConversationConstraint(ContractModel):
    """One structured fact extracted from KISC conversation context."""

    slot: NonEmptyString
    value: NonEmptyString
    polarity: Literal["positive", "negative", "uncertain"]
    source_turn_id: NonEmptyString


class FrameFeedback(ContractModel):
    """Cumulative human decisions about candidate frames."""

    accepted_frame_ids: list[NonEmptyString] = Field(default_factory=list)
    rejected_frame_ids: list[NonEmptyString] = Field(default_factory=list)

    @field_validator("accepted_frame_ids", "rejected_frame_ids")
    @classmethod
    def deduplicate_ids(cls, frame_ids: list[str]) -> list[str]:
        """Deduplicate frame IDs while preserving decision order."""
        return list(dict.fromkeys(frame_ids))

    @model_validator(mode="after")
    def validate_disjoint_decisions(self) -> Self:
        """Reject an ambiguous update that accepts and rejects one frame."""
        overlap = set(self.accepted_frame_ids) & set(self.rejected_frame_ids)
        if overlap:
            raise ValueError(
                "accepted and rejected frame IDs must be disjoint"
            )
        return self


class ConversationSession(ContractModel):
    """Active KISC session state."""

    session_id: NonEmptyString
    created_at: int = Field(ge=0, description="Timestamp in milliseconds.")
    problem_id: NonEmptyString | None = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    feedback: FrameFeedback = Field(default_factory=FrameFeedback)


class SubmissionResult(ContractModel):
    """Official competition submission code."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    frame_idx: int = Field(ge=0)
    submission_code: NonEmptyString

    @model_validator(mode="after")
    def validate_submission_code(self) -> Self:
        """Keep the public code coupled to official frame identifiers."""
        if self.submission_code != f"{self.video_id},{self.frame_idx}":
            raise ValueError("submission_code must equal 'video_id,frame_idx'")
        return self
