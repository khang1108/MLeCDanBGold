"""Strict DRES v2 wire models for the standalone mock server.

This module owns the supported client API payloads and their semantic
validation. It does not implement HTTP behavior, task resolution, or scoring.
Field aliases follow the official DRES Client OpenAPI contract.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DresWireModel(BaseModel):
    """Base for DRES payloads with strict types, aliases, and closed schemas."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )


class DresVerdict(str, Enum):
    """Verdict values returned by DRES; the mock does not determine them."""

    CORRECT = "CORRECT"
    WRONG = "WRONG"
    INDETERMINATE = "INDETERMINATE"
    UNDECIDABLE = "UNDECIDABLE"


class LoginRequest(DresWireModel):
    """Credentials accepted by the DRES login endpoint."""

    username: str
    password: str


class ApiUser(DresWireModel):
    """Optional user fields returned by the DRES client API."""

    id: str | None = None
    username: str | None = None
    role: Literal["ANYONE", "VIEWER", "PARTICIPANT", "JUDGE", "ADMIN"] | None = None
    session_id: str | None = Field(default=None, alias="sessionId")


class ApiClientTaskTemplateInfo(DresWireModel):
    """Active task template metadata; current-task responses have no task ID."""

    name: str
    task_group: str = Field(alias="taskGroup")
    task_type: str = Field(alias="taskType")
    duration: int | None = None


class ApiClientEvaluationInfo(DresWireModel):
    """Evaluation listing entry exposed to a DRES client."""

    id: str
    name: str
    type: Literal["SYNCHRONOUS", "ASYNCHRONOUS", "NON_INTERACTIVE"]
    status: Literal["CREATED", "ACTIVE", "TERMINATED"]
    template_id: str = Field(alias="templateId")
    template_description: str | None = Field(default=None, alias="templateDescription")
    teams: list[str]
    task_templates: list[ApiClientTaskTemplateInfo] = Field(alias="taskTemplates")


class ApiClientAnswer(DresWireModel):
    """One textual, media-item, or temporal media answer from a client."""

    text: str | None = None
    media_item_name: str | None = Field(default=None, alias="mediaItemName")
    media_item_collection_name: str | None = Field(
        default=None,
        alias="mediaItemCollectionName",
    )
    start: int | None = None
    end: int | None = None

    @model_validator(mode="after")
    def validate_answer_shape(self) -> ApiClientAnswer:
        """Require one supported answer form while retaining supplied values."""

        has_text = self.text is not None
        has_media_item = self.media_item_name is not None

        if has_text == has_media_item:
            raise ValueError("answer must contain either text or a media item")

        if has_text:
            if not self.text.strip():
                raise ValueError("text answer must not be blank")
            if self.media_item_collection_name is not None:
                raise ValueError("media collection requires a media item answer")
            if self.start is not None or self.end is not None:
                raise ValueError("temporal endpoints require a media item answer")
            return self

        if not self.media_item_name.strip():
            raise ValueError("media item name must not be blank")

        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be provided together")
        if self.start is not None and self.end is not None:
            if self.start < 0 or self.end < 0:
                raise ValueError("temporal endpoints must be non-negative")
            if self.end < self.start:
                raise ValueError("end must be greater than or equal to start")

        return self


class ApiClientAnswerSet(DresWireModel):
    """Answers associated with an optional DRES task ID or task name."""

    task_id: str | None = Field(default=None, alias="taskId")
    task_name: str | None = Field(default=None, alias="taskName")
    answers: list[ApiClientAnswer] = Field(min_length=1)


class ApiClientSubmission(DresWireModel):
    """One DRES submission containing one or more non-empty answer sets."""

    answer_sets: list[ApiClientAnswerSet] = Field(alias="answerSets", min_length=1)


class QueryEvent(DresWireModel):
    """A timestamped client interaction included in a query result log."""

    timestamp: int
    category: Literal[
        "TEXT",
        "IMAGE",
        "SKETCH",
        "FILTER",
        "BROWSING",
        "COOPERATION",
        "OTHER",
    ]
    event_type: str = Field(alias="type")
    value: str


class RankedAnswer(DresWireModel):
    """A result log entry; DRES requires the answer but not a rank."""

    answer: ApiClientAnswer
    rank: int | None = None


class QueryResultLog(DresWireModel):
    """Required DRES result-log fields with arrays allowed to be empty."""

    timestamp: int
    sort_type: str = Field(alias="sortType")
    result_set_availability: str = Field(alias="resultSetAvailability")
    results: list[RankedAnswer]
    events: list[QueryEvent]


class SuccessStatus(DresWireModel):
    """Standard successful DRES response body."""

    status: bool
    description: str


class ErrorStatus(DresWireModel):
    """Standard failed DRES response body."""

    status: bool
    description: str


class SuccessfulSubmissionsStatus(DresWireModel):
    """DRES submission response carrying an operator-selected verdict."""

    status: bool
    submission: DresVerdict
    description: str


class TaskControlRequest(DresWireModel):
    """Operator-selected task metadata for the local test-control API."""

    active: bool
    name: str
    task_group: str = Field(alias="taskGroup")
    task_type: str = Field(alias="taskType")
    duration: int | None = None

    @field_validator("name", "task_group", "task_type")
    @classmethod
    def validate_nonblank_task_metadata(cls, value: str) -> str:
        """Reject whitespace-only task metadata without normalizing valid input."""

        if not value.strip():
            raise ValueError("task metadata must not be blank")
        return value


class SubmissionScenarioRequest(DresWireModel):
    """One-shot response settings for the next accepted submission."""

    status_code: Literal[200, 202, 400, 401, 404, 412, 500] = Field(
        default=200,
        alias="statusCode",
    )
    verdict: DresVerdict = Field(
        default=DresVerdict.INDETERMINATE,
        strict=False,
    )
    delay_ms: int = Field(default=0, alias="delayMs", ge=0, le=30_000)
    malformed_body: bool = Field(default=False, alias="malformedBody")
    description: str = "mock accepted"


class ResultLogScenarioRequest(DresWireModel):
    """One-shot response settings for the next accepted result log."""

    status_code: Literal[200, 202, 400, 401, 404, 412, 500] = Field(
        default=200,
        alias="statusCode",
    )
    delay_ms: int = Field(default=0, alias="delayMs", ge=0, le=30_000)
    malformed_body: bool = Field(default=False, alias="malformedBody")
    description: str = "mock accepted"


__all__ = [
    "ApiClientAnswer",
    "ApiClientAnswerSet",
    "ApiClientEvaluationInfo",
    "ApiClientSubmission",
    "ApiClientTaskTemplateInfo",
    "ApiUser",
    "DresVerdict",
    "DresWireModel",
    "ErrorStatus",
    "LoginRequest",
    "QueryEvent",
    "QueryResultLog",
    "RankedAnswer",
    "ResultLogScenarioRequest",
    "SuccessStatus",
    "SuccessfulSubmissionsStatus",
    "SubmissionScenarioRequest",
    "TaskControlRequest",
]
