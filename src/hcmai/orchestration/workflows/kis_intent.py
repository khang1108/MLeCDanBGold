"""Deterministic KIS intent builder from ordered historical clue inputs.

This module converts an ordered sequence of client-provided KIS clues into a
structured `KISIntent` containing the combined query string and planned temporal events.
It maintains no server-side mutable state and does not perform LLM-based query rewrites
in this baseline phase.
"""

from __future__ import annotations

from collections.abc import Sequence

from hcmai.api.contracts.kis import KISIntent
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.temporal import plan_query_events


class KISIntentBuilder:
    """Builds deterministic KIS search intents from ordered participant clues."""

    def __init__(
        self,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        """Initialize the intent builder with an event cardinality constraint."""
        self.max_temporal_event_count = max_temporal_event_count

    def build(self, inputs: Sequence[str]) -> KISIntent:
        """Construct a KISIntent from ordered clues.

        Args:
            inputs: Ordered sequence of raw clue strings.

        Returns:
            A validated KISIntent with normalized inputs, combined query, and planned events.

        Raises:
            ValueError: If inputs are empty, contain blank text, or produce more
                temporal events than allowed.
        """
        normalized = [" ".join(text.split()) for text in inputs]
        if not normalized or any(not text for text in normalized):
            raise ValueError("KIS intent inputs must contain non-empty text")

        # Join with newlines so plan_query_events recognizes distinct clues
        # even when participants do not provide trailing punctuation.
        planned_text = "\n".join(normalized)
        planned_events = plan_query_events(planned_text)
        # Harmonize with sentence boundary splitting by stripping terminal punctuation.
        events = [
            stripped if (stripped := event.rstrip(".!?").rstrip()) else event
            for event in planned_events
        ]
        if len(events) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )

        query_text = " ".join(normalized)
        return KISIntent(
            revision=len(normalized),
            inputs=normalized,
            query_text=query_text,
            events=events,
        )
