"""Validated contracts for the optional DRES mini-challenge integration."""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, Field, model_validator

from hcmai.common.schemas.base import ContractModel, NonEmptyString


class _CamelCaseContract(ContractModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class MiniChallengeLoginRequest(ContractModel):
    username: NonEmptyString
    password: str


class MiniChallengeLoginResponse(_CamelCaseContract):
    id: str | None = None
    username: str
    role: str | None = None
    session_id: str = Field(alias="sessionId")


class MiniChallengeTaskTemplate(_CamelCaseContract):
    name: NonEmptyString
    task_group: NonEmptyString = Field(alias="taskGroup")
    task_type: NonEmptyString = Field(alias="taskType")
    duration: int | None = Field(default=None, ge=0)


class MiniChallengeEvaluation(_CamelCaseContract):
    id: NonEmptyString
    name: NonEmptyString
    status: str
    type: str | None = None
    template_id: str | None = Field(default=None, alias="templateId")
    template_description: str | None = Field(
        default=None, alias="templateDescription"
    )
    teams: list[str] = Field(default_factory=list)
    task_templates: list[MiniChallengeTaskTemplate] = Field(
        default_factory=list, alias="taskTemplates"
    )


class MiniChallengeSubmitRequest(ContractModel):
    """Browser request that keeps canonical identity as a frame ID."""

    frame_id: NonEmptyString
    task_name: NonEmptyString
    text: str | None = Field(default=None, max_length=2_000)


class MiniChallengeAnswer(_CamelCaseContract):
    media_item_name: NonEmptyString = Field(alias="mediaItemName")
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)
    text: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class MiniChallengeAnswerSet(_CamelCaseContract):
    task_name: NonEmptyString = Field(alias="taskName")
    answers: list[MiniChallengeAnswer] = Field(min_length=1)


class MiniChallengeSubmission(_CamelCaseContract):
    answer_sets: list[MiniChallengeAnswerSet] = Field(
        alias="answerSets", min_length=1
    )


class MiniChallengeSubmissionResult(ContractModel):
    model_config = ConfigDict(extra="ignore")

    status: bool
    submission: str
    description: str

