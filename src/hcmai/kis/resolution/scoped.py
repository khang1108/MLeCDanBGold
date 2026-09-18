"""Resolve event-scoped KIS patches without widening their authorization.

This module owns structured scoped model output and its server-side application. It
never changes unrelated events or the canonical entity table, and it does not own
natural-language initial resolution or global rewriting.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from hcmai.inference.clients.llm import LLMClient
from hcmai.kis.models import (
    EventId,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISTemporalEdge,
    NonBlank,
)
from hcmai.kis.parser import EventPatchInstruction
from hcmai.kis.resolution.initial import KISResolutionError
from hcmai.kis.resolution.prompts import build_kis_scoped_messages


class ScopedResolvedEvent(BaseModel):
    """Model-authored text and bindings for one explicitly granted event."""

    model_config = ConfigDict(extra="forbid")
    event_id: EventId
    text: NonBlank
    bindings: list[KISEntityBinding] = Field(default_factory=list)


class ScopedResolutionBatch(BaseModel):
    """Model output for exactly the event IDs authorized by a patch batch."""

    model_config = ConfigDict(extra="forbid")
    events: list[ScopedResolvedEvent] = Field(min_length=1)


def canonical_query_text(events: Sequence[KISEvent]) -> str | None:
    """Join non-empty event text in timeline order, or return ``None`` if absent."""
    text = " ".join(event.text for event in events if event.text)
    return text or None


def _edges_for(events: Sequence[KISEvent]) -> list[KISTemporalEdge]:
    """Build the complete adjacent temporal chain for an ordered event list."""
    return [
        KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}")
        for i in range(1, len(events))
    ]


def apply_scoped_resolutions(
    base: KISIntent | None,
    resolved: ScopedResolutionBatch,
    *,
    revision: int,
) -> KISIntent:
    """Apply only resolved event IDs while preserving all other canonical evidence."""
    if revision < 1:
        raise ValueError("KIS revision must be at least 1")

    resolved_by_id = {event.event_id: event for event in resolved.events}
    if len(resolved_by_id) != len(resolved.events):
        raise KISResolutionError("Scoped resolution contains duplicate event IDs")

    if base is None:
        expected_ids = [f"E{i + 1}" for i in range(len(resolved.events))]
        if list(resolved_by_id) != expected_ids:
            raise KISResolutionError(
                f"Initial scoped event IDs must be contiguous E1..En, got {list(resolved_by_id)}"
            )
        if any(event.bindings for event in resolved.events):
            raise KISResolutionError("Initial scoped resolution cannot contain bindings")
        events = [
            KISEvent(id=event.event_id, text=event.text, bindings=[])
            for event in resolved.events
        ]
        entities = []
    else:
        base_ids = [event.id for event in base.events]
        base_count = len(base.events)
        target_numbers = [int(event_id[1:]) for event_id in resolved_by_id]
        if any(number < 1 for number in target_numbers):
            raise KISResolutionError("Scoped resolution contains an invalid event ID")
        new_numbers = sorted(number for number in target_numbers if number > base_count)
        if new_numbers and new_numbers != list(range(base_count + 1, new_numbers[-1] + 1)):
            raise KISResolutionError("New scoped event IDs must extend the timeline contiguously")
        if any(number <= base_count and f"E{number}" not in base_ids for number in target_numbers):
            raise KISResolutionError("Scoped resolution targets an unknown event")

        events = []
        for event in base.events:
            patch = resolved_by_id.get(event.id)
            if patch is None:
                events.append(event)
                continue
            bindings = _validated_bindings(patch.bindings, base)
            events.append(event.model_copy(update={"text": patch.text, "bindings": bindings}))
        for number in new_numbers:
            patch = resolved_by_id[f"E{number}"]
            events.append(
                KISEvent(
                    id=f"E{number}",
                    text=patch.text,
                    bindings=_validated_bindings(patch.bindings, base),
                )
            )
        entities = base.entities

    query_text = canonical_query_text(events)

    try:
        return KISIntent(
            revision=revision,
            query_text=query_text,
            entities=entities,
            events=events,
            temporal_edges=_edges_for(events),
        )
    except ValueError as exc:
        raise KISResolutionError(f"Canonical scoped intent validation failed: {exc}") from exc


def _validated_bindings(
    bindings: Sequence[KISEntityBinding], base: KISIntent
) -> list[KISEntityBinding]:
    """Validate model bindings against the immutable base entity table."""
    known = {entity.id for entity in base.entities}
    for binding in bindings:
        if binding.entity_id not in known:
            raise KISResolutionError(f"Scoped binding references unknown entity: {binding.entity_id}")
        if not binding.role.strip():
            raise KISResolutionError("Scoped binding role must be non-blank")
    return list(bindings)


class KISScopedResolver:
    """Request and apply semantic updates restricted to explicitly named events."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialize the resolver with a structured-output LLM client."""
        self._llm = llm

    def resolve(
        self,
        base: KISIntent | None,
        instructions: Sequence[EventPatchInstruction],
    ) -> ScopedResolutionBatch:
        """Resolve a patch batch and reject any model output outside granted IDs."""
        if not instructions:
            raise ValueError("Scoped instructions must not be empty")
        target_ids = [instruction.event_id for instruction in instructions]
        if len(target_ids) != len(set(target_ids)):
            raise KISResolutionError("Scoped instructions contain duplicate event IDs")
        if base is not None and any(event_id not in {e.id for e in base.events} and int(event_id[1:]) <= len(base.events) for event_id in target_ids):
            raise KISResolutionError("Scoped instruction targets an unknown event")

        description = base.model_dump_json() if base is not None else "No base intent; create E1..Ek."
        messages = build_kis_scoped_messages(
            description, [(item.event_id, item.instruction) for item in instructions]
        )
        resolved = self._llm.generate_structured(messages, ScopedResolutionBatch)
        returned_ids = [event.event_id for event in resolved.events]
        if set(returned_ids) != set(target_ids) or len(returned_ids) != len(target_ids):
            raise KISResolutionError(
                f"Scoped model output IDs {returned_ids} do not exactly match targets {target_ids}"
            )
        # Applying here validates bindings and all canonical graph invariants when base exists or instructions form a complete initial chain.
        if base is not None or [item.event_id for item in instructions] == [f"E{i + 1}" for i in range(len(instructions))]:
            apply_scoped_resolutions(base, resolved, revision=1 if base is None else base.revision + 1)
        return resolved


__all__ = [
    "KISScopedResolver",
    "ScopedResolvedEvent",
    "ScopedResolutionBatch",
    "apply_scoped_resolutions",
    "canonical_query_text",
]
