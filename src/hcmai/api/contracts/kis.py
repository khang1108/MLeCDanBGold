"""Stateless semantic operations and revisioned KIS interaction contracts.

These models define explicit semantic operations (initial resolve, scoped patches,
global rewrite, search-only) and immutable revision progression without server-side
session state or raw clue history.
"""

from __future__ import annotations

from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import SearchResult
from hcmai.kis.models import EventId, KISIntent

class KISOperationSummary(BaseModel):
    """Execution metadata describing the accepted semantic operation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["search_only"] = "search_only"
    affected_event_ids: list[EventId] = Field(default_factory=list)


class KISSearchRequest(BaseModel):
    """Stateless search request with explicit base intent or query hypothesis session."""

    model_config = ConfigDict(extra="forbid")
    query_hypothesis_session_id: str | None = None
    base_intent: KISIntent | None = None
    expected_revision: int = Field(default=0, ge=0)
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require an active target and at least one retrieval evidence source."""
        if not self.query_hypothesis_session_id and self.base_intent is None:
            raise ValueError("KIS search requires query_hypothesis_session_id or base_intent")
        if not self.use_dense and not self.use_bm25:
            has_image = self.base_intent is not None and any(bool(e.images) for e in self.base_intent.events)
            if not has_image and not self.query_hypothesis_session_id:
                raise ValueError("at least one retrieval source or image evidence must be available")
        return self


class KISSearchResult(SearchResult):
    """Ranked KIS result carrying an opaque result identifier for EventTrail handoff."""

    result_id: str


class KISSearchResponse(BaseModel):
    """Response payload for a successful revisioned KIS search."""

    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    operation_summary: KISOperationSummary
    use_dense: bool
    use_bm25: bool
    results: list[KISSearchResult] = Field(default_factory=list)
    latency: SearchLatency
    evidence_snapshot_id: str | None = None
    query_hypothesis_session_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("results", mode="before")
    @classmethod
    def _coerce_results(cls, v: Any) -> Any:
        if isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, SearchResult) and not isinstance(item, KISSearchResult):
                    coerced.append(
                        KISSearchResult(
                            result_id=f"r_{uuid4().hex}",
                            **item.model_dump(),
                        )
                    )
                else:
                    coerced.append(item)
            return coerced
        return v
