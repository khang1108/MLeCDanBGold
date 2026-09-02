"""Private inference-service HTTP contracts with no HCMAI runtime ownership."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


_NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class _HTTPContract(BaseModel):
    """Reject extra fields on private inference requests and responses."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TextEmbeddingRequest(_HTTPContract):
    """Ordered text batch routed to one configured encoder."""

    source: Literal["visual", "text"] = "visual"
    texts: list[_NonEmptyString] = Field(min_length=1)


class BoundaryScoreResponse(_HTTPContract):
    """Per-frame scores returned by a private boundary model."""

    request_id: _NonEmptyString
    model: _NonEmptyString
    revision: str | None = None
    scores: list[float] = Field(min_length=1)
    latency_ms: float = Field(ge=0)


class RerankItem(_HTTPContract):
    """One caller-owned item and its relevance score."""

    item_id: _NonEmptyString
    score: float


class RerankResponse(_HTTPContract):
    """Ordered relevance scores returned by the private reranker."""

    model: _NonEmptyString
    revision: str | None = None
    items: list[RerankItem]
    latency_ms: float = Field(ge=0)


class QueryEventsRequest(_HTTPContract):
    """Ordered non-empty retrieval events for literal translation."""

    events: list[_NonEmptyString] = Field(min_length=1)


class QueryCandidatesRequest(QueryEventsRequest):
    """Ordered retrieval events with the frozen public candidate count."""

    candidate_count: Literal[5] = 5


class QueryTranslationResponse(_HTTPContract):
    """Ordered literal English translations from the hosted model."""

    events: list[_NonEmptyString] = Field(min_length=1)


class QueryCandidatesResponse(_HTTPContract):
    """Literal translations and exactly five aligned event bundles."""

    literal_en: list[_NonEmptyString] = Field(min_length=1)
    candidates: list[list[_NonEmptyString]] = Field(min_length=5, max_length=5)


__all__ = [
    "BoundaryScoreResponse",
    "QueryCandidatesRequest",
    "QueryCandidatesResponse",
    "QueryEventsRequest",
    "QueryTranslationResponse",
    "RerankItem",
    "RerankResponse",
    "TextEmbeddingRequest",
]
