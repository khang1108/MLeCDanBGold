"""Stateless conversational-search API contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .conversation import ConversationState, ConversationTurn, FrameFeedback
from .enum import TaskType
from .search import SearchFilters, SearchResponse


class KISCSearchRequest(ContractModel):
    """One complete browser-owned conversational search turn."""

    query_type: Literal[TaskType.KISC] = TaskType.KISC
    history: list[ConversationTurn] = Field(default_factory=list)
    current_message: NonEmptyString = Field(max_length=1_000)
    previous_state: ConversationState | None = None
    feedback: FrameFeedback = Field(default_factory=FrameFeedback)
    top_k: int = Field(default=20, ge=1, le=100)
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        ids = [turn.turn_id for turn in self.history]
        if len(ids) != len(set(ids)):
            raise ValueError("history turn_id values must be unique")
        timestamps = [turn.created_at for turn in self.history]
        if timestamps != sorted(timestamps):
            raise ValueError("history must be ordered by created_at")
        return self


class KISCSearchResponse(ContractModel):
    """Resolved state and ranked search output for one stateless turn."""

    interpreted_state: ConversationState
    resolution_latency_ms: int = Field(ge=0)
    search: SearchResponse
    warnings: list[NonEmptyString] = Field(default_factory=list)
