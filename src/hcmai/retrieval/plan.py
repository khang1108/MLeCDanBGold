"""Immutable event-aligned retrieval views for textual KIS execution.

Canonical, dense, and literal text retain the same server-owned event IDs.
This S0 plan does not load images or perform translation or model inference.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KISRetrievalEvent:
    """Retain canonical and retriever-facing text for one semantic event."""

    event_id: str
    canonical_text: str | None
    dense_text: str | None
    bm25_text: str | None
    image_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep S0 text-only and reject blank supplied retrieval views."""
        if self.image_refs != ():
            raise ValueError("S0 retrieval events do not support image refs")
        for text in (self.canonical_text, self.dense_text, self.bm25_text):
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise ValueError("supplied event text must be nonblank")


@dataclass(frozen=True, slots=True)
class KISRetrievalPlan:
    """Preserve one immutable row per event in exactly E1..En order."""

    events: tuple[KISRetrievalEvent, ...]

    def __post_init__(self) -> None:
        """Reject empty, mutable, or misaligned event collections."""
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a nonempty tuple")
        if any(not isinstance(event, KISRetrievalEvent) for event in self.events):
            raise ValueError("events must contain KISRetrievalEvent rows")
        if self.event_ids != tuple(f"E{i}" for i in range(1, len(self.events) + 1)):
            raise ValueError("event IDs must be sequential E1..En")

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Return the shared order of every retrieval view."""
        return tuple(event.event_id for event in self.events)

    @property
    def canonical_texts(self) -> tuple[str, ...] | None:
        """Return canonical rows only when the complete text view exists."""
        values = tuple(event.canonical_text for event in self.events)
        if any(value is None for value in values):
            return None
        return tuple(str(value) for value in values)

    @property
    def dense_texts(self) -> tuple[str, ...] | None:
        """Return dense rows only when every event has dense text."""
        values = tuple(event.dense_text for event in self.events)
        if any(value is None for value in values):
            return None
        return tuple(str(value) for value in values)

    @property
    def bm25_texts(self) -> tuple[str, ...] | None:
        """Return literal rows only when every event has BM25 text."""
        values = tuple(event.bm25_text for event in self.events)
        if any(value is None for value in values):
            return None
        return tuple(str(value) for value in values)

    def validate_text_sources(self, *, use_dense: bool, use_bm25: bool) -> None:
        """Require complete canonical and enabled scoring views for S0."""
        if not isinstance(use_dense, bool) or not isinstance(use_bm25, bool):
            raise ValueError("retrieval source flags must be booleans")
        if not use_dense and not use_bm25:
            raise ValueError("at least one retrieval source must be enabled")
        if self.canonical_texts is None:
            raise ValueError("S0 requires canonical text for every event")
        if use_dense and self.dense_texts is None:
            raise ValueError("every event requires dense text when Dense is enabled")
        if use_bm25 and self.bm25_texts is None:
            raise ValueError("every event requires BM25 text when BM25 is enabled")
