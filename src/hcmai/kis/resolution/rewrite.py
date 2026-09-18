"""Perform topology-preserving global KIS semantic rewrites.

The model may rewrite text, entities, and bindings, but the server owns event IDs,
ordering, images, temporal edges, and revision. Image-only events are never given
model-invented text.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from hcmai.inference.clients.llm import LLMClient
from hcmai.kis.models import (
    EventId,
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISTemporalEdge,
    NonBlank,
)
from hcmai.kis.resolution.initial import KISResolutionError
from hcmai.kis.resolution.prompts import build_kis_global_rewrite_messages
from hcmai.kis.resolution.scoped import canonical_query_text


class GlobalRewriteEvent(BaseModel):
    """Model-authored text and bindings for an existing event ID."""

    model_config = ConfigDict(extra="forbid")
    event_id: EventId
    text: NonBlank | None = None
    bindings: list[KISEntityBinding] = Field(default_factory=list)


class GlobalRewriteResolution(BaseModel):
    """Structured global rewrite response with no server-owned topology fields."""

    model_config = ConfigDict(extra="forbid")
    query_text: NonBlank | None
    entities: list[KISEntity] = Field(default_factory=list)
    events: list[GlobalRewriteEvent] = Field(min_length=1)


class KISGlobalRewriter:
    """Rewrite semantic content while retaining the base event topology."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialize the rewriter with a structured-output LLM client."""
        self._llm = llm

    def rewrite(self, base: KISIntent, instruction: str) -> KISIntent:
        """Apply one global rewrite and assign exactly ``base.revision + 1``."""
        if not instruction.strip():
            raise ValueError("Global rewrite instruction must not be blank")
        messages = build_kis_global_rewrite_messages(base.model_dump_json(), instruction)
        resolved = self._llm.generate_structured(messages, GlobalRewriteResolution)

        expected_ids = [event.id for event in base.events]
        returned_ids = [event.event_id for event in resolved.events]
        if returned_ids != expected_ids:
            raise KISResolutionError(
                f"Global rewrite must return event IDs in base order {expected_ids}, got {returned_ids}"
            )

        known_entities = {entity.id for entity in resolved.entities}
        events: list[KISEvent] = []
        for original, update in zip(base.events, resolved.events, strict=True):
            if original.text is None:
                if update.text is not None:
                    raise KISResolutionError(
                        f"Image-only event {original.id} cannot receive invented text"
                    )
            elif update.text is None:
                raise KISResolutionError(
                    f"Text-bearing event {original.id} cannot lose its text"
                )
            for binding in update.bindings:
                if binding.entity_id not in known_entities:
                    raise KISResolutionError(
                        f"Global binding references unknown entity: {binding.entity_id}"
                    )
                if not binding.role.strip():
                    raise KISResolutionError("Global binding role must be non-blank")
            events.append(
                original.model_copy(update={"text": update.text, "bindings": list(update.bindings)})
            )

        query_text = resolved.query_text or canonical_query_text(events)
        if canonical_query_text(events) is None:
            query_text = None

        temporal_edges = [
            KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}")
            for i in range(1, len(events))
        ]
        try:
            return KISIntent(
                revision=base.revision + 1,
                query_text=query_text,
                entities=resolved.entities,
                events=events,
                temporal_edges=temporal_edges,
            )
        except ValueError as exc:
            raise KISResolutionError(f"Canonical rewrite validation failed: {exc}") from exc


__all__ = [
    "GlobalRewriteEvent",
    "GlobalRewriteResolution",
    "KISGlobalRewriter",
]
