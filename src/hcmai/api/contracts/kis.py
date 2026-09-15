"""Revisioned KIS interaction contracts for one-shot and multi-clue search.

These models define client-owned ordered clue inputs and deterministic revision
semantics without introducing server-side mutable session state.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    model_validator,
)

from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import SearchResult
from hcmai.kis.models import EventId, KISIntent
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class KISInput(BaseModel):
    """One immutable clue or query fragment sent by the participant."""

    model_config = ConfigDict(extra="forbid")
    text: NonBlank


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


class KISExplorationEventSeed(BaseModel):
    """Transport one committed event's canonical and scoring text views."""

    model_config = ConfigDict(extra="forbid")
    event_id: EventId
    canonical_text: NonBlank | None = None
    dense_text: NonBlank | None = None
    bm25_text: NonBlank | None = None


class KISExplorationSeed(BaseModel):
    """Capture ordered scoring inputs for a committed semantic revision."""

    model_config = ConfigDict(extra="forbid")
    semantic_revision: int = Field(ge=1)
    events: list[KISExplorationEventSeed] = Field(min_length=1)
    use_dense: StrictBool
    use_bm25: StrictBool

    def to_plan(self) -> KISRetrievalPlan:
        """Copy transport rows into an immutable, event-aligned plan."""
        return KISRetrievalPlan(
            events=tuple(
                KISRetrievalEvent(**event.model_dump()) for event in self.events
            )
        )

    @model_validator(mode="after")
    def validate_text_snapshot(self) -> Self:
        """Reject event order and missing scoring views before opening S0 branches."""
        self.to_plan().validate_text_sources(
            use_dense=self.use_dense, use_bm25=self.use_bm25,
        )
        return self


class KISRevisionSearchResponse(BaseModel):
    """Response payload for a successful revisioned KIS search."""

    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    exploration_seed: KISExplorationSeed
    use_dense: bool
    use_bm25: bool
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
    warnings: list[str] = Field(default_factory=list)
