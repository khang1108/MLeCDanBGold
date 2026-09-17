"""Immutable event-aligned retrieval views for multimodal KIS execution.

Canonical, dense, literal text, and image exemplars retain the same server-owned
event IDs.
"""

from dataclasses import dataclass
from typing import Any

from hcmai.kis.models import KISImageRef


@dataclass(frozen=True, slots=True)
class KISRetrievalEvent:
    """Retain canonical and retriever-facing text and images for one semantic event."""

    event_id: str
    canonical_text: str | None
    dense_text: str | None
    bm25_text: str | None
    image_refs: tuple[KISImageRef, ...] = ()

    def __post_init__(self) -> None:
        """Reject blank supplied retrieval views and require text or image evidence."""
        if not isinstance(self.image_refs, tuple):
            raise ValueError("image_refs must be a tuple")
        if any(not isinstance(ref, KISImageRef) for ref in self.image_refs):
            raise ValueError("image_refs must contain KISImageRef instances")
        if self.canonical_text is None and not self.image_refs:
            raise ValueError("KIS event requires text or image evidence")
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
    def event_count(self) -> int:
        """Return the number of planned events."""
        return len(self.events)

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Return the shared order of every retrieval view."""
        return tuple(event.event_id for event in self.events)

    @property
    def image_ref_rows(self) -> tuple[tuple[KISImageRef, ...], ...]:
        """Return the sequence of image references per event."""
        return tuple(event.image_refs for event in self.events)

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
        """Require valid scoring views for text and image events."""
        if not isinstance(use_dense, bool) or not isinstance(use_bm25, bool):
            raise ValueError("retrieval source flags must be booleans")
        has_any_text = any(event.canonical_text is not None for event in self.events)
        has_any_images = any(len(event.image_refs) > 0 for event in self.events)
        if not has_any_text and not has_any_images:
            raise ValueError("at least one event must have text or image evidence")
        if has_any_text and not use_dense and not use_bm25 and not has_any_images:
            raise ValueError("at least one retrieval source must be enabled")
        for event in self.events:
            if event.canonical_text is not None:
                if not use_dense and not use_bm25 and not event.image_refs:
                    raise ValueError("text events require at least one enabled text source")
                if use_dense and event.dense_text is None:
                    raise ValueError("every event with text requires dense text when Dense is enabled")
                if use_bm25 and event.bm25_text is None:
                    raise ValueError("every event with text requires BM25 text when BM25 is enabled")
            elif not event.image_refs:
                raise ValueError("events without text require image evidence")


def build_retrieval_plan(
    intent: Any,
    overrides: dict[str, Any] | None = None,
    *,
    use_dense: bool = True,
    use_bm25: bool = True,
) -> KISRetrievalPlan:
    """Build an immutable retrieval plan from intent and optional retrieval overrides.

    Canonical text always originates from the intent event. Dense and BM25 views
    incorporate overrides when provided.
    """
    rows: list[KISRetrievalEvent] = []
    for event in intent.events:
        override = (overrides or {}).get(event.id)
        dense_text: str | None = None
        if use_dense:
            if override is not None and getattr(override, "dense_text", None) is not None:
                dense_text = override.dense_text
            elif isinstance(override, dict) and override.get("dense_text") is not None:
                dense_text = override["dense_text"]
            else:
                dense_text = event.text

        bm25_text: str | None = None
        if use_bm25:
            if override is not None and getattr(override, "bm25_text", None) is not None:
                bm25_text = override.bm25_text
            elif isinstance(override, dict) and override.get("bm25_text") is not None:
                bm25_text = override["bm25_text"]
            else:
                bm25_text = event.text

        rows.append(
            KISRetrievalEvent(
                event_id=event.id,
                canonical_text=event.text,
                dense_text=dense_text,
                bm25_text=bm25_text,
                image_refs=tuple(event.images),
            )
        )
    return KISRetrievalPlan(events=tuple(rows))
