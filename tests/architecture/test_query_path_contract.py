"""Architecture and regression contract tests for HCMAI query-to-path flow.

This test module verifies production implementations against hand-vetted contract fixtures:
1. FastAPI app schema surface: preserves the exact HTTP path set and request/response
   contracts for /api/v1/search, /api/v1/trake, and /api/v1/query-candidates via create_app().
2. Planner characterization: real plan_query_events() folds attribute sentences, drops
   trailing questions, and restores chronological timeline order.
3. Canonical path invariants: real SearchMaterializer validates aligned paths against
   canonical Corpus records, enforces same-video / coordinate invariants, and projects
   the upper-middle representative frame.
4. ASR boundary contract: real TranscriptStore & Corpus.transcript() enforce half-open
   interval [start_ms, end_ms).
5. Object evidence contract: real ObjectCountsStore distinguishes completed empty counts ({})
   from missing or failed evaluations (None), while Corpus.object_counts() provides safe fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hcmai.app import create_app
from hcmai.corpus import Corpus
from hcmai.corpus.stores import ObjectCountsStore
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.temporal.dp import AlignedPath
from hcmai.temporal.planner import plan_query_events
from offline.enrichment.models import ProcessingStatus
from offline.enrichment.transcripts.models import TranscriptSegment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "query_path_contract"


def _load_fixture(filename: str) -> dict:
    fixture_file = FIXTURES_DIR / filename
    if not fixture_file.exists():
        pytest.fail(f"Required fixture file not found: {fixture_file}")
    with open(fixture_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_contract_corpus(root: Path) -> tuple[Corpus, ObjectCountsStore]:
    """Create a minimal, non-GPU Corpus instance backed by tiny Parquet artifacts."""
    frames_path = root / "frames.parquet"
    pd.DataFrame(
        [
            # test_video_001 frames
            {
                "frame_id": "test_video_001_f00100",
                "video_id": "test_video_001",
                "frame_idx": 100,
                "timestamp_ms": 4000,
                "image_path": "keyframes/test_video_001/100.jpg",
                "thumbnail_path": "keyframes/test_video_001/thumb_100.jpg",
                "width": 1280,
                "height": 720,
            },
            {
                "frame_id": "test_video_001_f00150",
                "video_id": "test_video_001",
                "frame_idx": 150,
                "timestamp_ms": 6000,
                "image_path": "keyframes/test_video_001/150.jpg",
                "thumbnail_path": "keyframes/test_video_001/thumb_150.jpg",
                "width": 1280,
                "height": 720,
            },
            {
                "frame_id": "test_video_001_f00300",
                "video_id": "test_video_001",
                "frame_idx": 300,
                "timestamp_ms": 12000,
                "image_path": "keyframes/test_video_001/300.jpg",
                "thumbnail_path": "keyframes/test_video_001/thumb_300.jpg",
                "width": 1280,
                "height": 720,
            },
            # test_video_002 frames
            {
                "frame_id": "test_video_002_f00100",
                "video_id": "test_video_002",
                "frame_idx": 100,
                "timestamp_ms": 4000,
                "image_path": "keyframes/test_video_002/100.jpg",
                "thumbnail_path": "keyframes/test_video_002/thumb_100.jpg",
                "width": 1280,
                "height": 720,
            },
            {
                "frame_id": "test_video_002_f00200",
                "video_id": "test_video_002",
                "frame_idx": 200,
                "timestamp_ms": 8000,
                "image_path": "keyframes/test_video_002/200.jpg",
                "thumbnail_path": "keyframes/test_video_002/thumb_200.jpg",
                "width": 1280,
                "height": 720,
            },
            {
                "frame_id": "test_video_002_f00250",
                "video_id": "test_video_002",
                "frame_idx": 250,
                "timestamp_ms": 10000,
                "image_path": "keyframes/test_video_002/250.jpg",
                "thumbnail_path": "keyframes/test_video_002/thumb_250.jpg",
                "width": 1280,
                "height": 720,
            },
            {
                "frame_id": "test_video_002_f00400",
                "video_id": "test_video_002",
                "frame_idx": 400,
                "timestamp_ms": 16000,
                "image_path": "keyframes/test_video_002/400.jpg",
                "thumbnail_path": "keyframes/test_video_002/thumb_400.jpg",
                "width": 1280,
                "height": 720,
            },
        ]
    ).to_parquet(frames_path, index=False)

    # Transcript artifact for ASR interval testing
    transcripts_path = root / "transcripts.parquet"
    pd.DataFrame(
        [
            TranscriptSegment(
                segment_id="seg_asr_test_001",
                video_id="vid_asr_001",
                segment_index=0,
                start_ms=1500,
                end_ms=4500,
                text="lời thoại trong đoạn video",
                language="vi",
            ).model_dump(mode="json")
        ]
    ).to_parquet(transcripts_path, index=False)

    # Object counts artifact: detected, zero, and failed status
    object_counts_path = root / "object-counts.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "test_video_001_f00100",
                "video_id": "test_video_001",
                "frame_idx": 100,
                "timestamp_ms": 4000,
                "counts_json": json.dumps({"person": 3, "bicycle": 1}),
                "status": ProcessingStatus.COMPLETED.value,
            },
            {
                "frame_id": "test_video_001_f00150",
                "video_id": "test_video_001",
                "frame_idx": 150,
                "timestamp_ms": 6000,
                "counts_json": json.dumps({}),
                "status": ProcessingStatus.COMPLETED.value,
            },
            {
                "frame_id": "test_video_001_f00300",
                "video_id": "test_video_001",
                "frame_idx": 300,
                "timestamp_ms": 12000,
                "counts_json": json.dumps({}),
                "status": ProcessingStatus.FAILED.value,
            },

        ]
    ).to_parquet(object_counts_path, index=False)

    raw_store = ObjectCountsStore(object_counts_path)
    corpus = Corpus.open(
        frames_path=frames_path,
        evidence_paths={},
        dataset_root=root,
        object_counts_path=object_counts_path,
        transcript_path=transcripts_path,
    )
    return corpus, raw_store


# -----------------------------------------------------------------------------
# 1. API Golden Contract Surface
# -----------------------------------------------------------------------------

def test_api_schema_surface_matches_golden_contract() -> None:
    """FastAPI app must expose identical endpoint paths and schemas matching golden."""
    golden = _load_fixture("api_golden_contract.json")
    expected_endpoints = golden["endpoints"]

    app = create_app()
    openapi = app.openapi()
    paths = openapi.get("paths", {})
    actual_endpoints = {path: paths.get(path) for path in expected_endpoints}

    assert actual_endpoints == expected_endpoints
    assert openapi.get("components") == golden["components"]


# -----------------------------------------------------------------------------
# 2. Planner Real Implementation
# -----------------------------------------------------------------------------

def test_planner_fixtures_against_real_implementation() -> None:
    """Run real plan_query_events against planner contract fixtures."""
    fixtures = _load_fixture("planner_fixtures.json")

    # Attribute sentence folding
    attr = fixtures["attribute_sentence"]
    planned_attr = plan_query_events(attr["input_query"])
    assert len(planned_attr) == attr["expected_count"]
    assert planned_attr == tuple(attr["expected_events"])

    # Trailing question dropping
    qa = fixtures["trailing_question"]
    planned_qa = plan_query_events(qa["input_query"])
    assert len(planned_qa) == qa["expected_count"]
    assert planned_qa == tuple(qa["expected_events"])
    assert qa["forbidden_substring"] not in planned_qa[0]

    # 'Trước đó' timeline chronological order restoration
    order = fixtures["truoc_do_temporal_order"]
    planned_order = plan_query_events(order["input_query"])
    assert len(planned_order) == order["expected_count"]
    assert planned_order == tuple(order["expected_events"])
    assert planned_order[0].startswith(order["first_event_prefix"])
    assert planned_order[1].startswith(order["second_event_prefix"])


# -----------------------------------------------------------------------------
# 3. Path Validation & Projection through Real SearchMaterializer
# -----------------------------------------------------------------------------

def test_valid_paths_through_real_materializer(tmp_path: Path) -> None:
    """Real SearchMaterializer must correctly project valid AlignedPath instances."""
    corpus, _ = _build_contract_corpus(tmp_path)
    materializer = SearchMaterializer(corpus)
    path_fixtures = _load_fixture("path_fixtures.json")

    # Test singleton path (1 event -> midpoint 0)
    single_data = path_fixtures["singleton_path"]
    single_path = AlignedPath(
        video_id=single_data["video_id"],
        score=single_data["score"],
        frame_ids=tuple(single_data["frame_ids"]),
        frame_idxs=tuple(single_data["frame_idxs"]),
        timestamps_ms=tuple(single_data["timestamps_ms"]),
    )
    res_single = materializer.build_kis_result(single_path)
    assert res_single.frame_id == single_data["expected_kis_representative"]["frame_id"]
    assert res_single.frame_idx == single_data["expected_kis_representative"]["frame_idx"]
    assert res_single.timestamp_ms == single_data["expected_kis_representative"]["timestamp_ms"]
    assert res_single.score == single_data["score"]

    # Test 3-event path (3 events -> midpoint 3//2 = 1)
    three_data = path_fixtures["three_event_path"]
    three_path = AlignedPath(
        video_id=three_data["video_id"],
        score=three_data["score"],
        frame_ids=tuple(three_data["frame_ids"]),
        frame_idxs=tuple(three_data["frame_idxs"]),
        timestamps_ms=tuple(three_data["timestamps_ms"]),
    )
    res_three = materializer.build_kis_result(three_path)
    assert res_three.frame_id == three_data["expected_kis_representative"]["frame_id"]
    assert res_three.frame_idx == three_data["expected_kis_representative"]["frame_idx"]
    assert res_three.timestamp_ms == three_data["expected_kis_representative"]["timestamp_ms"]
    assert res_three.score == three_data["score"]
    assert res_three.frame_ids == list(three_data["frame_ids"])
    assert res_three.timestamps_ms == list(three_data["timestamps_ms"])


def test_invalid_paths_rejected_by_real_materializer(tmp_path: Path) -> None:
    """Real SearchMaterializer must reject paths with canonical identity or coordinate drift."""
    corpus, _ = _build_contract_corpus(tmp_path)
    materializer = SearchMaterializer(corpus)

    # 1. Cross-video mismatch: path.video_id is test_video_001, but frame belongs to test_video_002
    cross_video_path = AlignedPath(
        video_id="test_video_001",
        score=1.5,
        frame_ids=("test_video_002_f00200",),  # representative frame (index 0) has video_id test_video_002
        frame_idxs=(200,),
        timestamps_ms=(8000,),
    )
    with pytest.raises(ValueError, match="video_id"):
        materializer.build_kis_result(cross_video_path)

    # 2. Coordinate drift: frame_idx disagrees with canonical frame record
    idx_drift_path = AlignedPath(
        video_id="test_video_001",
        score=1.0,
        frame_ids=("test_video_001_f00100",),
        frame_idxs=(999,),  # canonical is 100
        timestamps_ms=(4000,),
    )
    with pytest.raises(ValueError, match="frame_idx"):
        materializer.build_kis_result(idx_drift_path)

    # 3. Coordinate drift: timestamp_ms disagrees with canonical frame record
    ts_drift_path = AlignedPath(
        video_id="test_video_001",
        score=1.0,
        frame_ids=("test_video_001_f00100",),
        frame_idxs=(100,),
        timestamps_ms=(99999,),  # canonical is 4000
    )
    with pytest.raises(ValueError, match="timestamp"):
        materializer.build_kis_result(ts_drift_path)

    # 4. Length mismatch across arrays
    length_mismatch_path = AlignedPath(
        video_id="test_video_001",
        score=1.0,
        frame_ids=("test_video_001_f00100", "test_video_001_f00150"),
        frame_idxs=(100,),  # length 1 vs 2
        timestamps_ms=(4000, 6000),
    )
    with pytest.raises(ValueError, match="equal lengths"):
        materializer.build_kis_result(length_mismatch_path)

    # 5. Empty path
    empty_path = AlignedPath(
        video_id="test_video_001",
        score=0.0,
        frame_ids=(),
        frame_idxs=(),
        timestamps_ms=(),
    )
    with pytest.raises(ValueError, match="at least one frame"):
        materializer.build_kis_result(empty_path)


# -----------------------------------------------------------------------------
# 4. ASR Boundary Probes against Real Corpus & TranscriptStore
# -----------------------------------------------------------------------------

def test_asr_boundary_probes_against_real_corpus(tmp_path: Path) -> None:
    """Verify half-open [start_ms, end_ms) boundary directly through real Corpus."""
    corpus, _ = _build_contract_corpus(tmp_path)
    evidence_fixtures = _load_fixture("evidence_fixtures.json")
    asr_fixture = evidence_fixtures["asr_boundary_contract"]
    seg_info = asr_fixture["segment"]
    video_id = seg_info["video_id"]
    expected_text = seg_info["text"]

    for probe in asr_fixture["boundary_probes"]:
        ts = probe["timestamp_ms"]
        expected_match = probe["expected_match"]

        # Call real Corpus.transcript() with the exact 1-ms interval used by SearchMaterializer
        actual_transcript = corpus.transcript(video_id, ts, ts + 1)
        actual_segments = corpus.transcript_segments(video_id, ts, ts + 1)

        if expected_match:
            assert actual_transcript == expected_text, (
                f"Timestamp {ts}ms should match ASR segment: {probe['reason']}"
            )
            assert len(actual_segments) == 1
            assert actual_segments[0].segment_id == seg_info["segment_id"]
        else:
            assert actual_transcript is None, (
                f"Timestamp {ts}ms should NOT match ASR segment: {probe['reason']}"
            )
            assert len(actual_segments) == 0


# -----------------------------------------------------------------------------
# 5. Object Evidence Missing vs Empty against Real Store & Corpus
# -----------------------------------------------------------------------------

def test_object_evidence_against_real_store_and_corpus(tmp_path: Path) -> None:
    """Verify ObjectCountsStore preserves None vs {}, and Corpus provides fallback."""
    corpus, raw_store = _build_contract_corpus(tmp_path)
    evidence_fixtures = _load_fixture("evidence_fixtures.json")
    scenarios = evidence_fixtures["object_evidence_contract"]["scenarios"]

    # Scenario 1: Detected objects present
    # - Store returns dict of counts
    # - Corpus.object_counts returns dict
    # - Corpus.objects returns sorted labels
    counts_detected = raw_store.get_counts("test_video_001_f00100")
    assert counts_detected == scenarios["detected_objects"]["expected_optional"]
    assert corpus.object_counts("test_video_001_f00100") == scenarios["detected_objects"]["expected_fallback"]
    assert corpus.objects("test_video_001_f00100") == ("bicycle", "person")

    # Scenario 2: Zero detected objects (COMPLETED with empty dict)
    # - Store MUST return {} (NOT None, since processing completed)
    # - Corpus.object_counts returns {}
    # - Corpus.objects returns ()
    counts_zero = raw_store.get_counts("test_video_001_f00150")
    assert counts_zero == scenarios["zero_detected_objects"]["expected_optional"]
    assert counts_zero == {}
    assert counts_zero is not None, "Completed evaluation with 0 objects must return {}, not None"
    assert corpus.object_counts("test_video_001_f00150") == scenarios["zero_detected_objects"]["expected_fallback"]
    assert corpus.objects("test_video_001_f00150") == ()

    # Scenario 3: Failed enrichment status
    # - Store MUST return None (evidence not reliably available)
    # - Corpus.object_counts falls back to {}
    counts_failed = raw_store.get_counts("test_video_001_f00300")
    assert counts_failed is None
    assert corpus.object_counts("test_video_001_f00300") == scenarios["failed_enrichment"]["expected_fallback"]
    assert corpus.objects("test_video_001_f00300") == ()

    # Scenario 4: Absent frame record
    # - Store MUST return None (record does not exist)
    # - Corpus.object_counts falls back to {}
    counts_absent = raw_store.get_counts("non_existent_frame")
    assert counts_absent is None
    assert corpus.object_counts("non_existent_frame") == scenarios["missing_frame_record"]["expected_fallback"]
    assert corpus.objects("non_existent_frame") == ()
