"""Domain resolver for semantic KIS intent graphs using an LLM.

This module owns transforming initial natural-language queries into a validated
KISIntent graph. It delegates semantic reasoning to an LLMClient producing a
bounded KISInitialResolution with source fragments, verifies source grounding
server-side, and safely falls back to a single unsegmented event on any
inference or grounding failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from hcmai.inference.clients.llm import LLMClient
from hcmai.inference.errors import (
    InferenceError,
    InferenceResponseError,
    InferenceUnavailableError,
)
from hcmai.kis.hypothesis.grounding import align_source_fragments, infer_query_language
from hcmai.kis.models import (
    KISEvent,
    KISInitialResolution,
    KISIntent,
    KISTemporalEdge,
    SourceProvenance,
)
from hcmai.kis.resolution.prompts import build_kis_initial_messages


class KISResolutionError(RuntimeError):
    """Raised when an LLM resolution violates semantic or referential integrity."""


DEFAULT_INITIAL_RESOLVER_MAX_TOKENS = 256


def _fallback_intent(query: str, revision: int) -> KISIntent:
    """Construct safe fallback intent preserving the untouched query as one grounded event."""
    return KISIntent(
        revision=revision,
        query_text=query,
        language=infer_query_language(query),
        entities=[],
        events=[
            KISEvent(
                id="E1",
                text=query,
                origin="source",
                source_provenance=SourceProvenance(
                    source_text=query, start_char=0, end_char=len(query)
                ),
                bindings=[],
            )
        ],
        temporal_edges=[],
    )


class KISIntentResolver:
    """Resolves initial natural-language KIS input into a semantic graph."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_tokens: int = DEFAULT_INITIAL_RESOLVER_MAX_TOKENS,
    ) -> None:
        """Initialize with a capability-level LLMClient."""
        self._llm = llm
        self._max_tokens = max_tokens

    def resolve_initial(self, text: str, revision: int) -> KISIntent:
        """Resolve one initial natural-language query at a server-owned revision.

        The returned intent is a semantic snapshot and deliberately does not
        retain raw client input history.
        """
        return self._resolve((text,), revision=revision)

    def resolve(self, inputs: Sequence[str], *, revision: int) -> KISIntent:
        """Resolve legacy ordered inputs while callers migrate to ``resolve_initial``.

        This compatibility entry point preserves existing runtime behavior without
        restoring raw inputs to the canonical semantic intent.
        """
        return self._resolve(inputs, revision=revision)

    def _resolve(self, inputs: Sequence[str], *, revision: int) -> KISIntent:
        """Canonicalize validated natural-language input using the configured LLM.

        Args:
            inputs: Ordered list of participant clues from revision 1 to current.

        Returns:
            A validated KISIntent graph with server-owned revision, canonical IDs,
            source provenance, and sequential temporal edges.

        Raises:
            ValueError: If inputs are empty or contain blank strings, or revision < 1.
        """
        normalized = tuple(" ".join(value.split()) for value in inputs)
        if not normalized or any(not value for value in normalized):
            raise ValueError("KIS inputs must contain non-empty text")
        if revision < 1:
            raise ValueError("KIS revision must be at least 1")

        canonical_query = " ".join(normalized)
        messages = build_kis_initial_messages(normalized)

        try:
            resolution = self._llm.generate_structured(
                messages,
                KISInitialResolution,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
            fragments = [
                event.source_text
                for event in resolution.events
                if event.source_text
            ]
            if not fragments:
                return _fallback_intent(canonical_query, revision)

            spans = align_source_fragments(canonical_query, fragments)
            events = [
                KISEvent(
                    id=f"E{index + 1}",
                    text=span.source_text,
                    source_provenance=span,
                    origin="source",
                    bindings=[],
                )
                for index, span in enumerate(spans)
            ]
            edges = [
                KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}")
                for i in range(1, len(events))
            ]
            return KISIntent(
                revision=revision,
                query_text=canonical_query,
                language=infer_query_language(canonical_query),
                entities=[],
                events=events,
                temporal_edges=edges,
            )
        except (
            InferenceError,
            InferenceResponseError,
            InferenceUnavailableError,
            ValueError,
        ):
            return _fallback_intent(canonical_query, revision)
