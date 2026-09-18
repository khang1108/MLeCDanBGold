"""Browser-safe private DRES session and task-aware direct submission contracts.

Only a participant ID, frozen task-scope key, and answer batch cross the API.
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


class VbsTaskTemplateItem(BaseModel):
    """One task template metadata item within an evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: NonBlank
    task_group: NonBlank
    task_type: NonBlank
    duration: int | None = Field(default=None, ge=0, strict=True)


class VbsEvaluationSummary(BaseModel):
    """Safe evaluation and task summary visible to a connected participant."""

    model_config = ConfigDict(extra="forbid")

    id: NonBlank
    name: NonBlank
    type: NonBlank
    status: Literal["CREATED", "ACTIVE", "TERMINATED"]
    template_id: NonBlank
    task_templates: list[VbsTaskTemplateItem]


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
    """Task-aware direct submission against the frozen task scope."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    expected_task_scope_key: NonBlank
    answers: list[VbsAnswer] = Field(default_factory=list)
    evaluation_id: NonBlank | None = None
    task_name: NonBlank | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_answers(cls, data: object) -> object:
        if isinstance(data, dict):
            if "answer" in data and "answers" in data:
                raise ValueError("Cannot provide both 'answer' and 'answers'")
            if "answer" in data:
                ans = data.get("answer")
                data = dict(data)
                del data["answer"]
                data["answers"] = [ans] if ans is not None else []
        return data

    @model_validator(mode="after")
    def _validate_answers_presence(self) -> "VbsDirectSubmissionRequest":
        if not self.answers:
            raise ValueError("Field required: answers must contain at least one item")
        return self

    @property
    def answer(self) -> VbsAnswer:
        """Backward compatibility for single-answer callers."""
        return self.answers[0]



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
    "VbsEvaluationSummary",
    "VbsSessionConnectRequest",
    "VbsSessionStatus",
    "VbsTaskResponse",
    "VbsTaskTemplateItem",
    "VbsTemporalAnswer",
    "VbsTextAnswer",
]
