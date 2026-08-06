"""Competition temporal retrieval and key-event alignment contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import TaskType

NonNegativeFrameIndex = Annotated[int, Field(ge=0)]


class TRAKERequest(ContractModel):
    """Raw TRAKE query with optional caller-supplied ordered events."""

    query_type: Literal[TaskType.TRAKE] = TaskType.TRAKE
    query: NonEmptyString
    events: list[NonEmptyString] | None = Field(default=None, min_length=2)
    top_k: int = Field(default=20, ge=1, le=100)


class TRAKESubmission(ContractModel):
    """One same-video row containing one canonical frame per ordered event."""

    rank: int = Field(ge=1, le=100)
    video_id: NonEmptyString
    frame_ids: list[NonEmptyString] = Field(min_length=2)
    frame_idxs: list[NonNegativeFrameIndex] = Field(min_length=2)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_frame_sequence(self) -> Self:
        if len(self.frame_ids) != len(self.frame_idxs):
            raise ValueError("frame_ids and frame_idxs must have equal lengths")
        if any(
            current < previous
            for previous, current in zip(self.frame_idxs, self.frame_idxs[1:])
        ):
            raise ValueError("frame_idxs must preserve event order")
        return self


class TRAKEResponse(ContractModel):
    """Ranked TRAKE paths for one parsed ordered event sequence."""

    request_id: NonEmptyString
    query_type: Literal[TaskType.TRAKE] = TaskType.TRAKE
    query: NonEmptyString
    events: list[NonEmptyString] = Field(min_length=2)
    top_k: int = Field(ge=1, le=100)
    total_results: int = Field(ge=0, le=100)
    submissions: list[TRAKESubmission] = Field(default_factory=list)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_submissions(self) -> Self:
        if self.total_results != len(self.submissions):
            raise ValueError("total_results must equal the number of submissions")
        if self.total_results > self.top_k:
            raise ValueError("total_results must not be greater than top_k")
        expected_ranks = list(range(1, self.total_results + 1))
        if [row.rank for row in self.submissions] != expected_ranks:
            raise ValueError("submission ranks must be consecutive and one-based")
        if any(len(row.frame_ids) != len(self.events) for row in self.submissions):
            raise ValueError("each submission must contain one frame per event")
        return self
