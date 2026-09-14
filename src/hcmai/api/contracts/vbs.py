"""Browser-safe private DRES session and one-answer submission contracts.

Only a participant ID, frozen task-scope key, and one answer cross the API.
DRES credentials, session values, workspace state, and attempt identifiers stay
outside these contracts.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DresVerdict = Literal["CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"]


class VbsSessionConnectRequest(BaseModel):
    """Connect the configured backend credential mapped to this user ID."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank


class VbsSessionStatus(BaseModel):
    """Return participant connectivity without exposing a DRES token."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    connected: bool


class VbsTaskResponse(BaseModel):
    """Expose safe active-task metadata and its opaque frozen-scope key."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    evaluation_id: NonBlank
    task_scope_key: NonBlank
    task_name: NonBlank
    task_group: NonBlank
    task_type: NonBlank
    duration: int | None = Field(default=None, ge=0, strict=True)


class VbsTemporalAnswer(BaseModel):
    """One canonical video interval expressed in competition milliseconds."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["TEMPORAL"]
    video_id: NonBlank
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def validate_range(self) -> "VbsTemporalAnswer":
        """Reject reversed intervals before they can reach the DRES client."""

        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be greater than or equal to start_ms")
        return self


class VbsTextAnswer(BaseModel):
    """One non-blank text answer for a VQA task."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["TEXT"]
    text: NonBlank


VbsAnswer = Annotated[
    VbsTemporalAnswer | VbsTextAnswer,
    Field(discriminator="kind"),
]


class VbsDirectSubmissionRequest(BaseModel):
    """Submit one answer against the task scope frozen by the open popup."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    expected_task_scope_key: NonBlank
    answer: VbsAnswer


class VbsDirectSubmissionRecorded(BaseModel):
    """A DRES 2xx delivery with its official judge verdict."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["RECORDED"]
    recorded: Literal[True]
    verdict: DresVerdict
    message: NonBlank


class VbsDirectSubmissionNotRecorded(BaseModel):
    """A definitive DRES rejection that the participant may review manually."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["NOT_RECORDED"]
    recorded: Literal[False]
    verdict: None
    reason: Literal["DRES_REJECTED", "DRES_AUTH_REJECTED"]
    message: NonBlank


class VbsDirectSubmissionUnknown(BaseModel):
    """An ambiguous transport outcome that must never be replayed silently."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["UNKNOWN"]
    recorded: None
    verdict: None
    message: NonBlank


VbsDirectSubmissionOutcome = Annotated[
    VbsDirectSubmissionRecorded
    | VbsDirectSubmissionNotRecorded
    | VbsDirectSubmissionUnknown,
    Field(discriminator="state"),
]


__all__ = [
    "VbsDirectSubmissionNotRecorded",
    "VbsDirectSubmissionOutcome",
    "VbsDirectSubmissionRecorded",
    "VbsDirectSubmissionRequest",
    "VbsDirectSubmissionUnknown",
    "VbsSessionConnectRequest",
    "VbsSessionStatus",
    "VbsTaskResponse",
    "VbsTemporalAnswer",
    "VbsTextAnswer",
]
