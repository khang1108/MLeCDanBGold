"""Tests for deterministic query hypothesis mutations and monotonic undo."""

import pytest

from hcmai.kis.hypothesis.models import (
    AddEvent,
    EditEvent,
    MergeEvents,
    ReorderEvents,
    SplitEvent,
)
from hcmai.kis.hypothesis.mutations import apply_query_action, restore_query_state
from hcmai.kis.models import (
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
    SourceProvenance,
)


@pytest.fixture
def base_intent() -> KISIntent:
    events = [
        KISEvent(
            id="E1",
            text="người đàn ông bước vào phòng",
            origin="source",
            source_provenance=SourceProvenance(
                source_text="người đàn ông bước vào phòng", start_char=0, end_char=27
            ),
            bindings=[],
        ),
        KISEvent(
            id="E2",
            text="ngồi xuống bàn",
            origin="source",
            source_provenance=SourceProvenance(
                source_text="ngồi xuống bàn", start_char=28, end_char=42
            ),
            bindings=[],
        ),
    ]
    edges = [KISTemporalEdge(source="E1", target="E2")]
    return KISIntent(
        revision=1,
        query_text="người đàn ông bước vào phòng ngồi xuống bàn",
        language="vi",
        entities=[],
        events=events,
        temporal_edges=edges,
    )


def test_split_recanonicalizes_ids_and_bumps_revision(base_intent: KISIntent) -> None:
    out = apply_query_action(
        base_intent, SplitEvent(event_id="E1", split_at=14, image_assignments={})
    )
    assert out.revision == base_intent.revision + 1
    assert [e.id for e in out.events] == ["E1", "E2", "E3"]
    assert [(x.source, x.target) for x in out.temporal_edges] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]


def test_merge_requires_adjacent_events(base_intent: KISIntent) -> None:
    three_events = apply_query_action(
        base_intent, AddEvent(position=2, text="uống nước")
    )
    try:
        apply_query_action(
            three_events, MergeEvents(left_event_id="E1", right_event_id="E3")
        )
    except ValueError as exc:
        assert "adjacent" in str(exc)
    else:
        raise AssertionError("non-adjacent merge must fail")


def test_merge_adjacent_combines_events_and_recanonicalizes(base_intent: KISIntent) -> None:
    out = apply_query_action(
        base_intent, MergeEvents(left_event_id="E1", right_event_id="E2")
    )
    assert out.revision == base_intent.revision + 1
    assert len(out.events) == 1
    assert out.events[0].id == "E1"
    assert "người đàn ông bước vào phòng" in out.events[0].text
    assert "ngồi xuống bàn" in out.events[0].text
    assert out.temporal_edges == []


def test_merge_source_events_preserves_exact_query_slice_provenance() -> None:
    query = "walk in,   then sit!"
    intent = KISIntent(
        revision=1,
        query_text=query,
        language="en",
        events=[
            KISEvent(
                id="E1",
                text="walk in,",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="walk in,", start_char=0, end_char=8
                ),
                bindings=[],
            ),
            KISEvent(
                id="E2",
                text="then sit!",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="then sit!", start_char=11, end_char=len(query)
                ),
                bindings=[],
            ),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2")],
    )

    merged = apply_query_action(
        intent, MergeEvents(left_event_id="E1", right_event_id="E2")
    ).events[0]

    assert merged.origin == "source"
    assert merged.text == query
    assert merged.source_provenance is not None
    assert merged.source_provenance.source_text == query
    assert merged.source_provenance.start_char == 0
    assert merged.source_provenance.end_char == len(query)


def test_merge_does_not_claim_source_when_query_span_cannot_be_verified() -> None:
    intent = KISIntent(
        revision=1,
        query_text="wrong then second",
        language="en",
        events=[
            KISEvent(
                id="E1",
                text="first",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="first", start_char=0, end_char=5
                ),
                bindings=[],
            ),
            KISEvent(
                id="E2",
                text="second",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="second", start_char=12, end_char=18
                ),
                bindings=[],
            ),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2")],
    )

    merged = apply_query_action(
        intent, MergeEvents(left_event_id="E1", right_event_id="E2")
    ).events[0]

    assert merged.origin == "user_override"
    assert merged.source_provenance is None


def test_reorder_recanonicalizes_events(base_intent: KISIntent) -> None:
    out = apply_query_action(
        base_intent, ReorderEvents(event_ids=("E2", "E1"))
    )
    assert out.revision == base_intent.revision + 1
    assert [e.id for e in out.events] == ["E1", "E2"]
    assert out.events[0].text == "ngồi xuống bàn"
    assert out.events[1].text == "người đàn ông bước vào phòng"
    assert [(x.source, x.target) for x in out.temporal_edges] == [("E1", "E2")]


def test_add_event_inserts_at_position(base_intent: KISIntent) -> None:
    out = apply_query_action(
        base_intent, AddEvent(position=1, text="cầm lấy tách trà")
    )
    assert out.revision == base_intent.revision + 1
    assert len(out.events) == 3
    assert [e.id for e in out.events] == ["E1", "E2", "E3"]
    assert out.events[1].text == "cầm lấy tách trà"
    assert out.events[1].origin == "user_added"
    assert out.events[1].source_provenance is None


def test_edit_preserves_provenance_and_marks_override(base_intent: KISIntent) -> None:
    out = apply_query_action(
        base_intent, EditEvent(event_id="E1", text="edited semantics")
    )
    assert out.events[0].origin == "user_override"
    assert out.events[0].source_provenance == base_intent.events[0].source_provenance


def test_undo_restore_is_monotonic(base_intent: KISIntent) -> None:
    edited = apply_query_action(base_intent, EditEvent(event_id="E1", text="edited"))
    restored = restore_query_state(edited, base_intent)
    assert restored.revision == edited.revision + 1
    assert restored.events[0].text == base_intent.events[0].text
