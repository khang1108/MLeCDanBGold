"""Strict Pydantic contracts for the small DRES v2 surface HCMAI uses.

This module owns DRES wire casing and validation, not HTTP transport or task
policy. Schemas reflect the official DRES Client OpenAPI 2.0.5-SNAPSHOT.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


# Source: https://raw.githubusercontent.com/dres-dev/DRES/master/doc/oas-client.json
# Observed DRES Client OpenAPI version: 2.0.5-SNAPSHOT.

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TaskText = Annotated[str, StringConstraints(min_length=1)]
DresVerdict = Literal["CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"]


class _DresModel(BaseModel):
    """Base DRES payload model that accepts Python names and emits aliases."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ApiClientAnswer(_DresModel):
    """One DRES plaintext or temporal media answer."""

    text: str | None = None
    media_item_name: str | None = Field(default=None, alias="mediaItemName")
    media_item_collection_name: str | None = Field(
        default=None,
        alias="mediaItemCollectionName",
    )
    start: int | None = None
    end: int | None = None


class ApiClientAnswerSet(_DresModel):
    """One answer set tied to a task name accepted by DRES v2."""

    task_name: NonBlank | None = Field(default=None, alias="taskName")
    answers: list[ApiClientAnswer]


class ApiClientSubmission(_DresModel):
    """Submission request containing one or more task answer sets."""

    answer_sets: list[ApiClientAnswerSet] = Field(alias="answerSets")


class QueryEvent(_DresModel):
    """One text, image, or filter interaction attached to a result log."""

    timestamp: int = Field(ge=0)
    category: Literal[
        "TEXT",
        "IMAGE",
        "SKETCH",
        "FILTER",
        "BROWSING",
        "COOPERATION",
        "OTHER",
    ]
    event_type: NonBlank = Field(alias="type")
    value: str


class RankedAnswer(_DresModel):
    """One complete result in its original ranking position."""

    answer: ApiClientAnswer
    rank: int = Field(ge=1)


class QueryResultLog(_DresModel):
    """Result log containing all returned answers and its triggering event."""

    timestamp: int = Field(ge=0)
    sort_type: str = Field(alias="sortType")
    result_set_availability: str = Field(alias="resultSetAvailability")
    results: list[RankedAnswer]
    events: list[QueryEvent]


class DresUser(_DresModel):
    """DRES login identity; `sessionId` is retained only by backend service."""

    id: str | None = None
    username: str | None = None
    role: str | None = None
    session_id: NonBlank = Field(alias="sessionId")


class DresTaskTemplate(_DresModel):
    """One task template exposed by an evaluation listing."""

    name: str
    task_group: str = Field(alias="taskGroup")
    task_type: str = Field(alias="taskType")
    duration: int | None = None


class DresEvaluation(_DresModel):
    """Evaluation summary used for unique ACTIVE-run selection."""

    id: NonBlank
    name: str
    type: str
    status: Literal["CREATED", "ACTIVE", "TERMINATED"]
    template_id: str = Field(alias="templateId")
    template_description: str | None = Field(default=None, alias="templateDescription")
    teams: list[str]
    task_templates: list[DresTaskTemplate] = Field(alias="taskTemplates")


class DresTaskTemplateInfo(_DresModel):
    """Official current-task response; it contains template fields, not an ID."""

    name: TaskText
    task_group: TaskText = Field(alias="taskGroup")
    task_type: TaskText = Field(alias="taskType")
    duration: int | None = Field(default=None, ge=0)

    @field_validator("name", "task_group", "task_type")
    @classmethod
    def require_nonblank_without_normalizing(cls, value: str) -> str:
        """Reject blank task metadata while preserving DRES text verbatim."""

        if not value.strip():
            raise ValueError("task values must not be blank")
        return value


class DresSubmissionStatus(_DresModel):
    """Verdict-bearing status returned by an accepted DRES submission."""

    status: bool
    submission: DresVerdict
    description: str


class DresTaskScope(BaseModel):
    """Backend-only workspace scope derived from official task metadata."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: NonBlank
    task_scope_key: NonBlank
    task_name: TaskText
    task_group: TaskText
    task_type: TaskText
    duration: int | None = Field(default=None, ge=0)

    @field_validator("task_name", "task_group", "task_type")
    @classmethod
    def require_nonblank_without_normalizing(cls, value: str) -> str:
        """Validate the template text without changing its canonical value."""

        if not value.strip():
            raise ValueError("task values must not be blank")
        return value


class DresStatus(_DresModel):
    """Success or error status body returned by DRES endpoints."""

    status: bool
    description: str


__all__ = [
    "ApiClientAnswer",
    "ApiClientAnswerSet",
    "ApiClientSubmission",
    "DresEvaluation",
    "DresSubmissionStatus",
    "DresStatus",
    "DresTaskScope",
    "DresTaskTemplateInfo",
    "DresTaskTemplate",
    "DresVerdict",
    "DresUser",
    "QueryEvent",
    "QueryResultLog",
    "RankedAnswer",
]
