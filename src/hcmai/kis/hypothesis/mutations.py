"""Deterministic mutation operations and monotonic undo for KIS Query Hypotheses."""

from __future__ import annotations

from hcmai.kis.hypothesis.models import (
    AddEvent,
    EditEvent,
    MergeEvents,
    QueryHypothesisAction,
    ReorderEvents,
    SplitEvent,
)
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    KISTemporalEdge,
    SourceProvenance,
)


def _canonicalize(intent: KISIntent, events: list[KISEvent]) -> KISIntent:
    """Re-canonicalize event IDs to E1..En, rebuild adjacent edges, and bump revision."""
    canonical = [
        event.model_copy(update={"id": f"E{i + 1}"})
        for i, event in enumerate(events)
    ]
    edges = [
        KISTemporalEdge(source=f"E{i}", target=f"E{i + 1}")
        for i in range(1, len(canonical))
    ]
    return intent.model_copy(
        update={
            "revision": intent.revision + 1,
            "events": canonical,
            "temporal_edges": edges,
        }
    )


def _apply_split(intent: KISIntent, action: SplitEvent) -> KISIntent:
    event_idx = next(
        (i for i, e in enumerate(intent.events) if e.id == action.event_id), None
    )
    if event_idx is None:
        raise ValueError(f"Event {action.event_id} not found in intent")

    event = intent.events[event_idx]
    if event.text is None:
        raise ValueError("Cannot split an image-only event")

    if not (0 < action.split_at < len(event.text)):
        raise ValueError(
            f"split_at {action.split_at} must be between 1 and {len(event.text) - 1}"
        )

    left_text = event.text[: action.split_at].strip()
    right_text = event.text[action.split_at :].strip()
    if not left_text or not right_text:
        raise ValueError("Split boundary produces empty event text")

    # Image assignment validation
    assignments = action.image_assignments or {}
    for img in event.images:
        sides = assignments.get(img.asset_id, ())
        if not sides:
            raise ValueError(
                f"Missing explicit child assignment for image asset {img.asset_id}"
            )

    left_images = [
        img for img in event.images if "left" in assignments.get(img.asset_id, ())
    ]
    right_images = [
        img for img in event.images if "right" in assignments.get(img.asset_id, ())
    ]

    left_prov: SourceProvenance | None = None
    right_prov: SourceProvenance | None = None
    if event.source_provenance is not None and intent.query_text is not None:
        ls = intent.query_text.find(
            left_text, event.source_provenance.start_char
        )
        if ls >= 0 and ls + len(left_text) <= event.source_provenance.end_char:
            left_prov = SourceProvenance(
                source_text=left_text, start_char=ls, end_char=ls + len(left_text)
            )
            rs = intent.query_text.find(right_text, ls + len(left_text))
            if (
                rs >= 0
                and rs + len(right_text) <= event.source_provenance.end_char
            ):
                right_prov = SourceProvenance(
                    source_text=right_text,
                    start_char=rs,
                    end_char=rs + len(right_text),
                )

    left_event = KISEvent(
        id=event.id,
        text=left_text,
        source_provenance=left_prov,
        origin=event.origin,
        images=left_images,
        bindings=[],
    )
    right_event = KISEvent(
        id=event.id,
        text=right_text,
        source_provenance=right_prov,
        origin=event.origin,
        images=right_images,
        bindings=[],
    )

    new_events = [
        *intent.events[:event_idx],
        left_event,
        right_event,
        *intent.events[event_idx + 1 :],
    ]
    return _canonicalize(intent, new_events)


def _apply_merge(intent: KISIntent, action: MergeEvents) -> KISIntent:
    left_idx = next(
        (i for i, e in enumerate(intent.events) if e.id == action.left_event_id),
        None,
    )
    right_idx = next(
        (
            i
            for i, e in enumerate(intent.events)
            if e.id == action.right_event_id
        ),
        None,
    )
    if left_idx is None or right_idx is None:
        raise ValueError("Events to merge not found in intent")

    if right_idx != left_idx + 1:
        raise ValueError("Merge requires adjacent events in chronological order")

    left = intent.events[left_idx]
    right = intent.events[right_idx]

    texts = [t for t in (left.text, right.text) if t]
    merged_text = " ".join(texts) if texts else None

    # Merge images deduplicated by asset_id
    seen_assets: set[str] = set()
    merged_images = []
    for img in (*left.images, *right.images):
        if img.asset_id not in seen_assets:
            seen_assets.add(img.asset_id)
            merged_images.append(img)

    merged_prov: SourceProvenance | None = None
    if (
        left.source_provenance is not None
        and right.source_provenance is not None
        and left.source_provenance.end_char <= right.source_provenance.start_char
        and merged_text is not None
    ):
        merged_prov = SourceProvenance(
            source_text=merged_text,
            start_char=left.source_provenance.start_char,
            end_char=right.source_provenance.end_char,
        )

    origin = (
        "source"
        if (
            left.origin == "source"
            and right.origin == "source"
            and merged_prov is not None
        )
        else "user_override"
    )

    merged_event = KISEvent(
        id=left.id,
        text=merged_text,
        source_provenance=merged_prov,
        origin=origin,
        images=merged_images,
        bindings=[],
    )

    new_events = [
        *intent.events[:left_idx],
        merged_event,
        *intent.events[right_idx + 1 :],
    ]
    return _canonicalize(intent, new_events)


def _apply_reorder(intent: KISIntent, action: ReorderEvents) -> KISIntent:
    current_ids = [e.id for e in intent.events]
    if len(action.event_ids) != len(current_ids) or set(
        action.event_ids
    ) != set(current_ids):
        raise ValueError(
            f"Reorder requires exactly the existing event-ID set {current_ids}, got {action.event_ids}"
        )

    event_map = {e.id: e for e in intent.events}
    reordered = [event_map[eid] for eid in action.event_ids]
    return _canonicalize(intent, reordered)


def _apply_edit(intent: KISIntent, action: EditEvent) -> KISIntent:
    clean_text = " ".join(action.text.split())
    if not clean_text:
        raise ValueError("Edited event text cannot be blank")

    event_idx = next(
        (i for i, e in enumerate(intent.events) if e.id == action.event_id), None
    )
    if event_idx is None:
        raise ValueError(f"Event {action.event_id} not found in intent")

    current = intent.events[event_idx]
    updated = current.model_copy(
        update={"text": clean_text, "origin": "user_override"}
    )
    new_events = [
        *intent.events[:event_idx],
        updated,
        *intent.events[event_idx + 1 :],
    ]
    return intent.model_copy(
        update={"revision": intent.revision + 1, "events": new_events}
    )


def _apply_add(intent: KISIntent, action: AddEvent) -> KISIntent:
    if not (0 <= action.position <= len(intent.events)):
        raise ValueError(
            f"Position {action.position} out of bounds for event count {len(intent.events)}"
        )

    clean_text = " ".join(action.text.split()) if action.text else None
    if clean_text is None and not action.images:
        raise ValueError("Added event requires text or image evidence")

    new_event = KISEvent(
        id=f"E{len(intent.events) + 1}",
        text=clean_text,
        origin="user_added",
        source_provenance=None,
        images=list(action.images),
        bindings=[],
    )
    new_events = [
        *intent.events[: action.position],
        new_event,
        *intent.events[action.position :],
    ]
    return _canonicalize(intent, new_events)


def apply_query_action(
    intent: KISIntent, action: QueryHypothesisAction
) -> KISIntent:
    """Apply a deterministic structural or semantic action to a query hypothesis intent.

    Args:
        intent: Current canonical KISIntent.
        action: One of SplitEvent, MergeEvents, ReorderEvents, EditEvent, AddEvent.

    Returns:
        New KISIntent with monotonic revision and canonicalized IDs/edges.
    """
    if isinstance(action, SplitEvent):
        return _apply_split(intent, action)
    if isinstance(action, MergeEvents):
        return _apply_merge(intent, action)
    if isinstance(action, ReorderEvents):
        return _apply_reorder(intent, action)
    if isinstance(action, EditEvent):
        return _apply_edit(intent, action)
    if isinstance(action, AddEvent):
        return _apply_add(intent, action)
    raise TypeError(f"Unsupported query hypothesis action: {type(action).__name__}")


def restore_query_state(current: KISIntent, previous: KISIntent) -> KISIntent:
    """Restore an earlier semantic intent state with a strictly increasing revision.

    Monotonic undo invariant: revision never decrements.
    """
    return previous.model_copy(update={"revision": current.revision + 1})
