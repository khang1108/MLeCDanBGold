"""Explicit event text normalization for temporal queries and alignment.

This module owns deterministic normalization of ordered event sequences. It does
not perform semantic parsing, query planning, or model inference.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["normalize_event_texts"]


def normalize_event_texts(events: Sequence[str]) -> tuple[str, ...]:
    """Normalize a complete explicit event sequence without changing its alignment."""
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise ValueError("events must be a non-string sequence")
    if not events:
        raise ValueError("events must not be empty")
    if any(not isinstance(event, str) for event in events):
        raise ValueError("events must contain strings")

    normalized = tuple(" ".join(event.split()) for event in events)
    if any(not event for event in normalized):
        raise ValueError("events must contain non-empty strings")
    return normalized
