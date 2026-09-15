"""Domain resolver for semantic KIS intent graphs using an LLM.

This module owns transforming initial natural-language queries into a validated
KISIntent graph. It delegates language reasoning to an LLMClient producing a
semantic KISResolution, then canonicalizes IDs, revision, bindings, and temporal
chains server-side.
"""

from __future__ import annotations

from collections.abc import Sequence

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.inference.llm import LLMClient
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISResolution,
    KISTemporalEdge,
)
from hcmai.kis.prompts import build_kis_intent_messages


class KISResolutionError(RuntimeError):
    """Raised when an LLM resolution violates semantic or referential integrity."""


class KISIntentResolver:
    """Resolves initial natural-language KIS input into a semantic graph."""

    def __init__(self, llm: LLMClient) -> None:
        """Initialize with a capability-level LLMClient."""
        self._llm = llm

    def resolve_initial(self, query: str, *, revision: int) -> KISIntent:
        """Resolve one initial natural-language query at a server-owned revision.

        The returned intent is a semantic snapshot and deliberately does not
        retain raw client input history.
        """
        return self._resolve((query,), revision=revision)

    def resolve(self, inputs: Sequence[str]) -> KISIntent:
        """Resolve legacy ordered inputs while callers migrate to ``resolve_initial``.

        This compatibility entry point preserves existing runtime behavior without
        restoring raw inputs to the canonical semantic intent.
        """
        return self._resolve(inputs, revision=len(inputs))

    def _resolve(self, inputs: Sequence[str], *, revision: int) -> KISIntent:
        """Canonicalize validated natural-language input using the configured LLM.

        Args:
            inputs: Ordered list of participant clues from revision 1 to current.

        Returns:
            A validated KISIntent graph with server-owned revision, canonical IDs,
            and sequential temporal edges.

        Raises:
            ValueError: If inputs are empty or contain blank strings.
            KISResolutionError: If LLM output violates referential integrity or bounds.
        """
        normalized = tuple(" ".join(value.split()) for value in inputs)
        if not normalized or any(not value for value in normalized):
            raise ValueError("KIS inputs must contain non-empty text")
        if revision < 1:
            raise ValueError("KIS revision must be at least 1")

        messages = build_kis_intent_messages(normalized)
        resolution = self._llm.generate_structured(messages, KISResolution)

        if len(resolution.events) > DEFAULT_MAX_TEMPORAL_EVENT_COUNT:
            raise KISResolutionError(
                f"Resolution event count ({len(resolution.events)}) exceeds maximum allowed "
                f"({DEFAULT_MAX_TEMPORAL_EVENT_COUNT})"
            )

        entities = [
            KISEntity(id=f"X{i+1}", kind=e.kind, description=e.description)
            for i, e in enumerate(resolution.entities)
        ]

        events = []
        for event_index, event in enumerate(resolution.events):
            if len(set(event.entity_indices)) != len(event.entity_indices):
                raise KISResolutionError("event contains duplicate entity indices")
            bindings = []
            for entity_index in event.entity_indices:
                if not 0 <= entity_index < len(entities):
                    raise KISResolutionError(
                        f"event references an out-of-range entity index: {entity_index}"
                    )
                bindings.append(
                    KISEntityBinding(
                        entity_id=entities[entity_index].id,
                        role="participant",
                    )
                )
            events.append(KISEvent(id=f"E{event_index+1}", text=event.text, bindings=bindings))

        edges = [
            KISTemporalEdge(source=f"E{i}", target=f"E{i+1}")
            for i in range(1, len(events))
        ]

        try:
            return KISIntent(
                revision=revision,
                language=resolution.language,
                query_text=resolution.query_text,
                entities=entities,
                events=events,
                temporal_edges=edges,
            )
        except ValueError as exc:
            raise KISResolutionError(f"Canonical intent validation failed: {exc}") from exc
