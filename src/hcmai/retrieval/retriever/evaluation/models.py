"""Evaluation-query artifact contracts owned by the retrieval benchmark."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryDifficulty(str, Enum):
    """Human-assigned difficulty of an evaluation query."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationQuery(BaseModel):
    """One labelled query loaded by the offline evaluation harness."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(default="1.0", min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=1_000)
    language: Literal["vi", "en", "mixed"]
    task_type: Literal["kis", "trake"]
    difficulty: QueryDifficulty
    gold_frame_ids: list[str] = Field(min_length=1)
    gold_video_ids: list[str] = Field(default_factory=list)
    temporal_tolerance_ms: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("gold_frame_ids", "gold_video_ids", "tags")
    @classmethod
    def deduplicate_string_lists(cls, values: list[str]) -> list[str]:
        """Remove duplicates while retaining deterministic order."""

        if any(not value.strip() for value in values):
            raise ValueError("evaluation identifiers and tags must be non-empty")
        return list(dict.fromkeys(values))


__all__ = ["EvaluationQuery", "QueryDifficulty"]
