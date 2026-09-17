"""Stateless semantic operations and revisioned KIS interaction contracts.

These models define explicit semantic operations (initial resolve, scoped patches,
global rewrite, search-only) and immutable revision progression without server-side
session state or raw clue history.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import SearchResult
from hcmai.kis.models import EventId, KISImageRef, KISIntent

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

    def _request_may_have_image_evidence(self) -> bool:
        if self.base_intent is not None and any(event.images for event in self.base_intent.events):
            return True
        operation = self.operation
        if operation.kind == "initial_resolve":
            return bool(operation.image_refs) or any(patch.add_image_ids for patch in operation.patches)
        if operation.kind == "patch_events":
            return any(patch.add_image_ids for patch in operation.patches)
        return False

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        """Require at least one retrieval evidence source or image evidence."""
        if not self.use_dense and not self.use_bm25 and not self._request_may_have_image_evidence():
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
