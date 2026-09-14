"""Revisioned KIS interaction contracts for one-shot and multi-clue search.

These models define client-owned ordered clue inputs and deterministic revision
semantics without introducing server-side mutable session state.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import SearchResult

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KISInput(BaseModel):
    """One immutable clue or query fragment sent by the participant."""

    model_config = ConfigDict(extra="forbid")
    text: NonBlank


class KISIntent(BaseModel):
    """Deterministic interpretation of an ordered sequence of KIS clues."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    inputs: list[NonBlank] = Field(min_length=1)
    query_text: NonBlank
    events: list[NonBlank] = Field(min_length=1)


class KISRevisionSearchRequest(BaseModel):
    """Request for revisioned KIS search containing all historical clues."""

    model_config = ConfigDict(extra="forbid")
    inputs: list[KISInput] = Field(min_length=1)
    expected_revision: int = Field(ge=0)
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

    @property
    def previous_revision(self) -> int:
        """The revision index that this append or re-search expects as base."""
        return len(self.inputs) - 1

    @property
    def has_revision_conflict(self) -> bool:
        """Flag whether the client's expected base revision matches inputs length."""
        return self.expected_revision != self.previous_revision

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require at least one retrieval evidence source."""
        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one of use_dense or use_bm25 must be true")
        return self


class KISRevisionSearchResponse(BaseModel):
    """Response payload for a successful revisioned KIS search."""

    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    inputs: list[KISInput] = Field(min_length=1)
    intent: KISIntent
    query: str
    events: list[str]
    dense_events: list[str] | None = None
    bm25_caption_events: list[str] | None = None
    use_dense: bool
    use_bm25: bool
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
