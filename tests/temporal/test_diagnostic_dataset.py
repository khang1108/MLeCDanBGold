"""Tests for Transition Diagnostic Dataset schema, loader, and counterfactuals (SP-12)."""

from pathlib import Path
import pytest

from hcmai.temporal.diagnostic import (
    DiagnosticDataset,
    DiagnosticEvent,
    DiagnosticQuery,
)


def test_load_sample_diagnostic_dataset():
    """Verify data/eval/transition_diagnostic.json loads successfully with valid categories."""
    path = Path("data/eval/transition_diagnostic.json")
    assert path.is_file(), "data/eval/transition_diagnostic.json must exist"

    dataset = DiagnosticDataset.load_json(path)
    assert len(dataset) >= 6

    expected_categories = {
        "state_transition",
        "object_manipulation",
        "direction_change",
        "object_state_change",
        "multi_step_interaction",
        "order_permutation",
    }
    found_categories = {q.category for q in dataset.queries}
    assert expected_categories.issubset(found_categories)

    # Verify event counts are 2 to 4
    for q in dataset.queries:
        assert 2 <= len(q.events) <= 4
        for ev in q.events:
            assert ev.id.startswith("E")
            assert len(ev.text) > 0


def test_conversion_to_event_annotations():
    """Verify diagnostic dataset converts cleanly to candidate recall annotations."""
    dataset = DiagnosticDataset.load_json("data/eval/transition_diagnostic.json")
    annotations = dataset.to_event_annotations()

    total_events = sum(len(q.events) for q in dataset.queries)
    assert len(annotations) == total_events

    # Verify first annotation corresponds to DIAG_001 event 0
    ann0 = annotations[0]
    assert ann0.video_id == "V_DIAG_01"
    assert ann0.event_index == 0
    assert ann0.target_frame_idxs == (10, 20)
    assert ann0.start_ms == 1000
    assert ann0.end_ms == 3000


def test_counterfactual_reverse_and_swap():
    """Verify generation of reversed and adjacent-swap counterfactual queries."""
    ev1 = DiagnosticEvent(id="E1", text="sit", target_frame_idxs=(1,))
    ev2 = DiagnosticEvent(id="E2", text="stand", target_frame_idxs=(2,))
    ev3 = DiagnosticEvent(id="E3", text="walk", target_frame_idxs=(3,))

    query = DiagnosticQuery(
        query_id="Q_TEST",
        video_id="V_TEST",
        query="sit then stand then walk",
        category="motion",
        events=(ev1, ev2, ev3),
    )

    # 1. Reversed query
    rev = query.reversed_query()
    assert rev.query_id == "Q_TEST_reverse"
    assert [e.id for e in rev.events] == ["E3", "E2", "E1"]
    assert rev.events[0].text == "walk"
    assert rev.events[2].text == "sit"

    # 2. Adjacent swap (index 0 swaps E1 and E2)
    swapped = query.swapped_query(swap_index=0)
    assert swapped.query_id == "Q_TEST_swap_0"
    assert [e.id for e in swapped.events] == ["E2", "E1", "E3"]
    assert swapped.events[0].text == "stand"
    assert swapped.events[1].text == "sit"


def test_dataset_json_roundtrip(tmp_path):
    """Verify JSON export and re-import preserves all fields."""
    ev1 = DiagnosticEvent(id="E1", text="e1", start_ms=100, end_ms=200, target_frame_idxs=(1, 2))
    query = DiagnosticQuery(query_id="Q1", video_id="V1", query="e1", category="test", events=(ev1,))
    dataset = DiagnosticDataset(queries=(query,))

    out_file = tmp_path / "diag.json"
    dataset.save_json(out_file)

    loaded = DiagnosticDataset.load_json(out_file)
    assert len(loaded) == 1
    assert loaded.queries[0].query_id == "Q1"
    assert loaded.queries[0].events[0].target_frame_idxs == (1, 2)
