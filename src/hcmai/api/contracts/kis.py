"""Stateless semantic operations and revisioned KIS interaction contracts.

These models define explicit semantic operations (initial resolve, scoped patches,
global rewrite, search-only) and immutable revision progression without server-side
session state or raw clue history.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

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
from hcmai.kis.models import EventId, KISImageRef, KISIntent
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EventPatch(BaseModel):
    """Instruction and image delta for one named event in a patch batch."""

    model_config = ConfigDict(extra="forbid")
    event_id: EventId
    instruction: str | None = None
    add_image_ids: list[str] = Field(default_factory=list)
    remove_image_ids: list[str] = Field(default_factory=list)


class InitialResolveOperation(BaseModel):
    """Initial resolution via natural text, image refs, or explicit E# batch."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["initial_resolve"] = "initial_resolve"
    text: NonBlank | None = None
    image_refs: list[KISImageRef] = Field(default_factory=list)
    patches: list[EventPatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_initial_route(self) -> Self:
        natural_route = self.text is not None or bool(self.image_refs)
        explicit_route = bool(self.patches)
        if natural_route == explicit_route:
            raise ValueError(
                "initial_resolve requires exactly one natural/image route or explicit E# batch"
            )
        return self


class PatchEventsOperation(BaseModel):
    """Scoped semantic and image updates targeting existing or next contiguous events."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["patch_events"] = "patch_events"
    patches: list[EventPatch] = Field(min_length=1)


class GlobalRewriteOperation(BaseModel):
    """Rewrite text/entities across all events while preserving topology and images."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["global_rewrite"] = "global_rewrite"
    instruction: NonBlank


class SearchOnlyOperation(BaseModel):
    """Rerun retrieval over unchanged semantic intent with updated retrieval options."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["search_only"] = "search_only"


KISOperation = Annotated[
    InitialResolveOperation
    | PatchEventsOperation
    | GlobalRewriteOperation
    | SearchOnlyOperation,
    Field(discriminator="kind"),
]


class KISOperationSummary(BaseModel):
    """Execution metadata describing the accepted semantic operation."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["initial_resolve", "patch_events", "global_rewrite", "search_only"]
    affected_event_ids: list[EventId] = Field(default_factory=list)


class KISSearchRequest(BaseModel):
    """Stateless search request with explicit base intent and semantic operation."""

    model_config = ConfigDict(extra="forbid")
    base_intent: KISIntent | None = None
    expected_revision: int = Field(ge=0)
    operation: KISOperation
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)

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
        plan = self.to_plan()
        if plan.canonical_texts is not None:
            plan.validate_text_sources(
                use_dense=self.use_dense, use_bm25=self.use_bm25,
            )
        return self


class KISSearchResponse(BaseModel):
    """Response payload for a successful revisioned KIS search."""

    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    operation_summary: KISOperationSummary
    exploration_seed: KISExplorationSeed
    use_dense: bool
    use_bm25: bool
    results: list[SearchResult] = Field(default_factory=list)
    latency: SearchLatency
    warnings: list[str] = Field(default_factory=list)
