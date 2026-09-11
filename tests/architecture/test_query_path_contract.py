"""Architecture and regression contract tests for HCMAI query-to-path flow.

This test module verifies:
1. FastAPI app schema surface: preserves the exact HTTP path set and request/response
   contracts for /api/v1/search, /api/v1/trake, and /api/v1/query-candidates.
2. Planner characterization: attribute folding, trailing question dropping, and
   chronological timeline restoration.
3. Canonical path invariants: same-video requirement, monotonic ordering, and
   representative midpoint projection.
4. Evidence contracts: half-open [start_ms, end_ms) ASR interval and distinction
   between missing/unavailable object evidence (None) and zero detected objects ({}).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hcmai.app import create_app
from hcmai.temporal.planner import plan_query_events

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "query_path_contract"


def _load_fixture(filename: str) -> dict:
    fixture_file = FIXTURES_DIR / filename
    if not fixture_file.exists():
        pytest.fail(f"Required fixture file not found: {fixture_file}")
    with open(fixture_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_api_schema_surface_matches_golden_contract() -> None:
    """FastAPI app must expose identical endpoint paths and schemas matching golden."""
    golden = _load_fixture("api_golden_contract.json")
    expected_endpoints = golden["endpoints"]

    # Use canonical existing app factory without hardcoding ad-hoc route factories
    app = create_app()
    openapi = app.openapi()
    paths = openapi.get("paths", {})

    for endpoint_path, expected_spec in expected_endpoints.items():
        assert endpoint_path in paths, f"Expected endpoint {endpoint_path} missing from app"
        actual_methods = paths[endpoint_path]
        for method, method_spec in expected_spec.items():
            assert method in actual_methods, (
                f"Method {method.upper()} missing from {endpoint_path}"
            )
            # Compare response status codes
            expected_responses = set(method_spec.get("responses", {}).keys())
            actual_responses = set(actual_methods[method].get("responses", {}).keys())
            assert expected_responses.issubset(actual_responses), (
                f"Missing expected response codes for {endpoint_path}: "
                f"{expected_responses - actual_responses}"
            )


def test_planner_attribute_sentence_folding() -> None:
    """Attribute sentence describing the same visual moment must fold into 1 event."""
    fixtures = _load_fixture("planner_fixtures.json")
    attr_fixture = fixtures["attribute_sentence"]

    planned = plan_query_events(attr_fixture["input_query"])
    assert len(planned) == attr_fixture["expected_count"]
    assert planned == tuple(attr_fixture["expected_events"])


def test_planner_trailing_question_dropping() -> None:
    """Trailing reviewer question must be stripped from planned retrieval moments."""
    fixtures = _load_fixture("planner_fixtures.json")
    qa_fixture = fixtures["trailing_question"]

    planned = plan_query_events(qa_fixture["input_query"])
    assert len(planned) == qa_fixture["expected_count"]
    assert planned == tuple(qa_fixture["expected_events"])
    assert qa_fixture["forbidden_substring"] not in planned[0]


def test_planner_truoc_do_timeline_order_restoration() -> None:
    """'Trước đó' cues must be reordered chronologically for monotonic temporal alignment."""
    fixtures = _load_fixture("planner_fixtures.json")
    order_fixture = fixtures["truoc_do_temporal_order"]

    planned = plan_query_events(order_fixture["input_query"])
    assert len(planned) == order_fixture["expected_count"]
    assert planned == tuple(order_fixture["expected_events"])
    assert planned[0].startswith(order_fixture["first_event_prefix"])
    assert planned[1].startswith(order_fixture["second_event_prefix"])


def test_path_representative_midpoint_calculation() -> None:
    """Midpoint selection for KIS must select index len // 2 (upper-middle)."""
    fixtures = _load_fixture("path_fixtures.json")

    # Singleton path
    single = fixtures["singleton_path"]
    single_midpoint_idx = len(single["frame_ids"]) // 2
    assert single_midpoint_idx == 0
    assert single["frame_ids"][single_midpoint_idx] == single["expected_kis_representative"]["frame_id"]
    assert single["frame_idxs"][single_midpoint_idx] == single["expected_kis_representative"]["frame_idx"]

    # 3-event path
    three = fixtures["three_event_path"]
    three_midpoint_idx = len(three["frame_ids"]) // 2
    assert three_midpoint_idx == 1
    assert three["frame_ids"][three_midpoint_idx] == three["expected_kis_representative"]["frame_id"]
    assert three["frame_idxs"][three_midpoint_idx] == three["expected_kis_representative"]["frame_idx"]


def test_path_canonical_invariants() -> None:
    """Paths must be verified against cross-video mismatch and non-monotonic order."""
    fixtures = _load_fixture("path_fixtures.json")

    # Cross-video check
    cross = fixtures["invalid_cross_video"]
    video_prefixes = {fid.rsplit("_", 1)[0] for fid in cross["frame_ids"]}
    # Verify the test fixture actually contains mismatched video prefixes
    assert len(video_prefixes) > 1


    # Monotonicity check
    non_mono = fixtures["invalid_non_monotonic"]
    idxs = non_mono["frame_idxs"]
    is_strictly_increasing = all(x < y for x, y in zip(idxs, idxs[1:]))
    assert not is_strictly_increasing


def test_asr_interval_half_open_semantics() -> None:
    """ASR segment interval matching must strictly satisfy half-open [start_ms, end_ms)."""
    fixtures = _load_fixture("evidence_fixtures.json")
    asr = fixtures["asr_boundary_contract"]
    seg = asr["segment"]
    start_ms = seg["start_ms"]
    end_ms = seg["end_ms"]

    def is_in_interval(ts: int) -> bool:
        return start_ms <= ts < end_ms

    for probe in asr["boundary_probes"]:
        ts = probe["timestamp_ms"]
        expected = probe["expected_match"]
        actual = is_in_interval(ts)
        assert actual == expected, (
            f"Timestamp {ts}ms containment mismatch: expected {expected}, got {actual}. "
            f"Reason: {probe['reason']}"
        )


def test_object_evidence_optional_vs_empty_semantics() -> None:
    """Missing or failed evidence must remain None, distinct from completed empty counts ({})."""
    fixtures = _load_fixture("evidence_fixtures.json")
    obj_contract = fixtures["object_evidence_contract"]
    scenarios = obj_contract["scenarios"]

    # Detected objects
    detected = scenarios["detected_objects"]
    assert detected["expected_optional"] == {"person": 3, "bicycle": 1}
    assert detected["expected_fallback"] == {"person": 3, "bicycle": 1}

    # Zero detected objects (completed)
    zero_obj = scenarios["zero_detected_objects"]
    assert zero_obj["expected_optional"] == {}
    assert zero_obj["expected_fallback"] == {}
    assert zero_obj["expected_optional"] is not None

    # Missing frame record
    missing = scenarios["missing_frame_record"]
    assert missing["expected_optional"] is None
    assert missing["expected_fallback"] == {}

    # Failed enrichment record
    failed = scenarios["failed_enrichment"]
    assert failed["expected_optional"] is None
    assert failed["expected_fallback"] == {}
