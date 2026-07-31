"""Contracts for optional AI-assisted query suggestions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .enum import QueryLanguage


class QuerySuggestionRequest(ContractModel):
    """User-authored query for which suggestions are requested."""

    query: NonEmptyString = Field(max_length=1_000)
    count: int | None = Field(default=None, ge=5, le=10)


class QuerySuggestionInferenceRequest(ContractModel):
    """Internal request sent to the configured model provider."""

    request_id: NonEmptyString
    query: NonEmptyString = Field(max_length=1_000)
    count: int = Field(ge=5, le=10)


class QuerySuggestion(ContractModel):
    """One standalone query which the operator may choose to search."""

    suggestion_id: NonEmptyString
    query: NonEmptyString = Field(max_length=1_000)
    language: QueryLanguage
    focus: Literal["literal", "action", "subject", "object", "scene", "temporal"]


class QuerySuggestionResponse(ContractModel):
    """Provider-independent suggestion response exposed to the frontend."""

    request_id: NonEmptyString
    original_query: NonEmptyString
    suggestions: list[QuerySuggestion] = Field(min_length=1, max_length=10)
    provider: Literal["gpu_inference", "openai_compatible"]
    model: NonEmptyString
    revision: str | None = None
    generation_latency_ms: float = Field(ge=0)
    warnings: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_queries(self) -> QuerySuggestionResponse:
        normalized = [" ".join(item.query.lower().split()) for item in self.suggestions]
        if len(normalized) != len(set(normalized)):
            raise ValueError("suggestion queries must be unique")
        if " ".join(self.original_query.lower().split()) in normalized:
            raise ValueError("suggestions must not repeat the original query")
        return self
