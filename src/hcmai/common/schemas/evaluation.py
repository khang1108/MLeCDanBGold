from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from .base import ContractModel, NonEmptyString
from .enum import QueryDifficulty, QueryLanguage

class EvaluationQuery(ContractModel):
    """One labelled query used by the offline evaluation harness."""

    schema_version: NonEmptyString = "1.0"
    query_id: NonEmptyString
    query: NonEmptyString = Field(max_length=1_000)
    language: QueryLanguage
    task_type: Literal["kis", "trake"]
    difficulty: QueryDifficulty
    gold_frame_ids: list[NonEmptyString] = Field(min_length=1)
    gold_video_ids: list[NonEmptyString] = Field(default_factory=list)
    temporal_tolerance_ms: int = Field(default=0, ge=0)
    tags: list[NonEmptyString] = Field(default_factory=list)
    notes: NonEmptyString | None = None

    @field_validator("gold_frame_ids", "gold_video_ids", "tags")
    @classmethod
    def deduplicate_string_lists(cls, values: list[str]) -> list[str]:
        """Remove duplicate strings while retaining deterministic order."""

        return list(dict.fromkeys(values))
