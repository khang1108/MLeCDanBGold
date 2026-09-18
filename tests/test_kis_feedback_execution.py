"""Tests for KIS chat feedback query repair and retrieval refinement execution."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackStateResponse,
    FeedbackTurnRequest,
    FeedbackUndoRequest,
    RetrievalOverride,
)
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.api.contracts.search import SearchResult
from hcmai.kis.feedback.models import (
    EditIntentAction,
    FeedbackCheckpoint,
    FeedbackResolveContext,
    FeedbackSession,
    RefineRetrievalAction,
    RejectCandidateAction,
    RestructureAction,
)
from hcmai.kis.feedback.resolver import FeedbackResolver
from hcmai.kis.feedback.service import FeedbackService
from hcmai.kis.feedback.store import FeedbackSessionStore
from hcmai.kis.models import (
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.retrieval.plan import build_retrieval_plan


def _create_sample_intent_with_image() -> KISIntent:
    img = KISImageRef(asset_id="asset_123", content_type="image/jpeg")
    return KISIntent(
        revision=1,
        query_text="A person enters and turns on a stove.",
        events=[
            KISEvent(id="E1", text="A person enters.", images=[img]),
            KISEvent(id="E2", text="A person turns on a stove."),
        ],
        temporal_edges=[
            KISTemporalEdge(source="E1", target="E2", relation="before"),
        ],
    )


from hcmai.api.contracts.search import SearchResult, SearchResultMetadata


def _make_search_result(
    result_id: str,
    video_id: str,
    frame_idx: int,
    timestamp_ms: int,
    score: float = 0.9,
) -> KISSearchResult:
    return KISSearchResult(
        result_id=result_id,
        frame_id=f"{video_id}_{frame_idx}",
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        score=score,
        frame_ids=[f"{video_id}_{frame_idx}"],
        timestamps_ms=[timestamp_ms],
        metadata=SearchResultMetadata(),
    )


class StubSearchExecution:
    def __init__(self, results: list[KISSearchResult], snapshot_id: str = "snap_test"):
        self.results = results
        self.evidence_snapshot_id = snapshot_id


class StubSearchService:
    def __init__(self, results_fn=None):
        self._results_fn = results_fn or self._default_results
        self.searches_run: list[dict] = []
        self.event_trail_snapshots = Mock()
        self.event_trail_snapshots.get.return_value = Mock(
            snapshot_id="snap_test",
            results={"r1": Mock(result_id="r1", video_id="video_001")},
        )

    def _default_results(self, plan):
        return [
            _make_search_result(f"r_{uuid4().hex[:8]}", "video_001", 10, 2000, 0.9),
            _make_search_result(f"r_{uuid4().hex[:8]}", "video_002", 25, 3000, 0.85),
        ]

    def execute_search(self, intent, retrieval_plan, use_dense=True, use_bm25=True, top_k=20, exclusions=None):
        self.searches_run.append({
            "intent": intent,
            "plan": retrieval_plan,
            "exclusions": exclusions,
        })
        results = self._results_fn(retrieval_plan)
        # Apply candidate exclusion filter if any
        if exclusions:
            filtered = []
            for r in results:
                # exclusion tuple: (video_id, event_id, frame_id_or_idx)
                if not any(r.video_id == ex[0] and str(r.frame_idx) == str(ex[2]) for ex in exclusions):
                    filtered.append(r)
            results = filtered
        return StubSearchExecution(results=results)


def test_refinement_preserves_semantic_intent():
    """Retrieval refinement updates retrieval override text, not canonical semantic intent."""
    store = FeedbackSessionStore()
    search_service = StubSearchService()
    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "A tall woman in a black jacket enters"},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    intent = _create_sample_intent_with_image()
    opened = service.open(
        FeedbackOpenRequest(
            intent=intent,
            original_query="A person enters and turns on a stove.",
            evidence_snapshot_id="snap_test",
        )
    )

    turn_req = FeedbackTurnRequest(
        request_id="req_1",
        expected_feedback_revision=opened.feedback_revision,
        expected_kis_revision=opened.intent.revision,
        message="Focus on tall woman in black jacket",
    )
    state = service.turn(opened.session_id, turn_req)

    # Intent unchanged, semantic revision unchanged
    assert state.intent.revision == 1
    assert state.intent.events[0].text == "A person enters."
    # Retrieval overrides updated
    assert "E1" in state.retrieval_overrides
    assert state.retrieval_overrides["E1"].dense_text == "A tall woman in a black jacket enters"
    assert state.feedback_revision == opened.feedback_revision + 1


def test_single_event_query_searches_full_corpus_and_can_change_top_video():
    """Single-event query search searches globally and can change top video."""
    store = FeedbackSessionStore()
    
    # Custom results function where refined query favors video_002
    def custom_results(plan):
        if plan.events[0].dense_text and "festival" in plan.events[0].dense_text:
            return [
                _make_search_result("r2", "video_002", 50, 2500, 0.99),
            ]
        return [
            _make_search_result("r1", "video_001", 10, 2000, 0.88),
        ]

    search_service = StubSearchService(results_fn=custom_results)
    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "Two people holding festival banner"},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    single_intent = KISIntent(
        revision=1,
        query_text="Two people holding banner",
        events=[KISEvent(id="E1", text="Two people holding banner")],
        temporal_edges=[],
    )
    opened = service.open(
        FeedbackOpenRequest(
            intent=single_intent,
            original_query="Two people holding banner",
            evidence_snapshot_id="snap_test",
        )
    )

    state = service.turn(
        opened.session_id,
        FeedbackTurnRequest(
            request_id="req_single",
            expected_feedback_revision=opened.feedback_revision,
            expected_kis_revision=opened.intent.revision,
            message="Banner có chữ festival",
        ),
    )

    assert state.scope == "all_videos"
    assert len(state.results) > 0
    assert state.results[0].video_id == "video_002"


def test_rejection_removes_only_rejected_occurrence_not_entire_video():
    """Rejecting a candidate removes only that occurrence without dropping other occurrences of the video."""
    store = FeedbackSessionStore()

    def multi_occurrence_results(plan):
        return [
            _make_search_result("r1", "video_001", 10, 1500, 0.95),
            _make_search_result("r2", "video_001", 20, 6500, 0.85),
        ]

    search_service = StubSearchService(results_fn=multi_occurrence_results)
    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RejectCandidateAction(
        event_id="E1",
        candidate_frame_id="10",
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    single_intent = KISIntent(
        revision=1,
        query_text="A car driving fast",
        events=[KISEvent(id="E1", text="A car driving fast")],
        temporal_edges=[],
    )
    opened = service.open(
        FeedbackOpenRequest(
            intent=single_intent,
            original_query="A car driving fast",
            evidence_snapshot_id="snap_test",
        )
    )

    state = service.turn(
        opened.session_id,
        FeedbackTurnRequest(
            request_id="req_rej",
            expected_feedback_revision=opened.feedback_revision,
            expected_kis_revision=opened.intent.revision,
            message="Frame này không phải",
            selected_result_id="r1",
            selected_event_id="E1",
            selected_frame_id="10",
        ),
    )

    # Frame 10 rejected, but video_001 frame 20 survives!
    assert len(state.results) == 1
    assert state.results[0].video_id == "video_001"
    assert state.results[0].frame_idx == 20


def test_restructure_split_creates_canonical_edges_and_preserves_images():
    """Restructure replaces contiguous block, rebuilds E1..En edges, and preserves unchanged images."""
    store = FeedbackSessionStore()
    search_service = StubSearchService()
    resolver = Mock(spec=FeedbackResolver)
    # Split E2 into 2 steps, preserving E1 (which has an image)
    resolver.resolve.return_value = RestructureAction(
        replaced_event_ids=["E2"],
        new_events=["Pours oil into pan.", "Lights the stove."],
        mapping={"E2": ["1", "2"]},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    intent = _create_sample_intent_with_image()
    opened = service.open(
        FeedbackOpenRequest(
            intent=intent,
            original_query="A person enters and turns on a stove.",
            evidence_snapshot_id="snap_test",
        )
    )

    state = service.turn(
        opened.session_id,
        FeedbackTurnRequest(
            request_id="req_split",
            expected_feedback_revision=opened.feedback_revision,
            expected_kis_revision=opened.intent.revision,
            message="Tách bước 2 thành: đổ dầu rồi bật bếp",
        ),
    )

    # Must have 3 events E1, E2, E3
    assert len(state.intent.events) == 3
    assert [ev.id for ev in state.intent.events] == ["E1", "E2", "E3"]
    # E1 image preserved
    assert len(state.intent.events[0].images) == 1
    assert state.intent.events[0].images[0].asset_id == "asset_123"
    # Canonical sequential temporal edges
    assert len(state.intent.temporal_edges) == 2
    assert state.intent.temporal_edges[0].source == "E1"
    assert state.intent.temporal_edges[0].target == "E2"
    assert state.intent.temporal_edges[1].source == "E2"
    assert state.intent.temporal_edges[1].target == "E3"
    # Revision bumped
    assert state.intent.revision == 2


def test_retrieval_error_rollback_preserves_active_state():
    """An error during retrieval execution must not mutate the active session state."""
    store = FeedbackSessionStore()
    search_service = Mock()
    search_service.event_trail_snapshots = Mock()
    search_service.event_trail_snapshots.get.return_value = Mock(snapshot_id="snap_test")
    search_service.execute_search.side_effect = RuntimeError("Retrieval gateway timed out")

    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "Crashing query"},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    intent = _create_sample_intent_with_image()
    opened = service.open(
        FeedbackOpenRequest(
            intent=intent,
            original_query="A person enters and turns on a stove.",
            evidence_snapshot_id="snap_test",
        )
    )

    with pytest.raises(RuntimeError):
        service.turn(
            opened.session_id,
            FeedbackTurnRequest(
                request_id="req_err",
                expected_feedback_revision=opened.feedback_revision,
                expected_kis_revision=opened.intent.revision,
                message="Break retrieval",
            ),
        )

    # Session in store remains at revision 1 and overrides empty
    current = store.get(opened.session_id)
    assert current.feedback_revision == opened.feedback_revision
    assert current.state.retrieval_overrides == {}


def test_duplicate_turn_submit_idempotency():
    """Submitting duplicate request_id with same payload returns cached response without rerunning search."""
    store = FeedbackSessionStore()
    search_service = StubSearchService()
    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "Idempotent refine"},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    intent = _create_sample_intent_with_image()
    opened = service.open(
        FeedbackOpenRequest(
            intent=intent,
            original_query="A person enters and turns on a stove.",
            evidence_snapshot_id="snap_test",
        )
    )

    turn_req = FeedbackTurnRequest(
        request_id="req_idem_1",
        expected_feedback_revision=opened.feedback_revision,
        expected_kis_revision=opened.intent.revision,
        message="Refine once",
    )
    first_res = service.turn(opened.session_id, turn_req)
    assert len(search_service.searches_run) == 1

    # Second call with same request_id
    second_res = service.turn(opened.session_id, turn_req)
    assert len(search_service.searches_run) == 1
    assert second_res.feedback_revision == first_res.feedback_revision


def test_feedback_execution_snapshots_temporal_evidence_for_event_trail():
    """Feedback turns delegate snapshot creation to search_service and preserve snapshot ID."""
    store = FeedbackSessionStore()
    search_service = Mock(spec=[])
    snapshot_store = Mock()
    snapshot_store.get.return_value = Mock(
        snapshot_id="snap_test",
        results={"r_init_1": Mock(result_id="r_init_1", video_id="video_001")},
    )
    search_service.event_trail_snapshots = snapshot_store

    # Mock real SearchPipeline.create_evidence_snapshot behavior
    snapshot_calls = []

    def mock_create_evidence_snapshot(intent, execution, result_ids):
        snapshot_calls.append((intent, execution, result_ids))
        return "snap_feedback_created_123", 42.5, []

    search_service.create_evidence_snapshot = mock_create_evidence_snapshot
    resolver = Mock(spec=FeedbackResolver)
    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "Refined query for event 1"},
    )
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    intent = _create_sample_intent_with_image()
    initial_res = [
        _make_search_result("r_init_1", "video_001", 10, 2000, 0.9),
    ]
    opened = service.open(
        FeedbackOpenRequest(
            intent=intent,
            original_query="A person enters and turns on a stove.",
            evidence_snapshot_id="snap_test",
            initial_results=initial_res,
        )
    )
    assert len(opened.results) == 1
    assert opened.results[0].result_id == "r_init_1"

    turn_req = FeedbackTurnRequest(
        request_id="req_snap_1",
        expected_feedback_revision=opened.feedback_revision,
        expected_kis_revision=opened.intent.revision,
        message="Refine with new visual keywords",
    )
    mock_kis_workflow = Mock()
    mock_kis_workflow.execute.return_value = Mock(
        results=[Mock(model_dump=lambda: {"video_id": "video_001", "frame_idx": 10, "timestamp_ms": 2000, "score": 0.95, "frame_id": "video_001_10", "frame_ids": ["video_001_10"], "timestamps_ms": [2000], "metadata": {}})],
        temporal_artifact=Mock(),
        latency=Mock(model_dump=lambda: {"total_ms": 100.0, "snapshot_ms": 0.0}),
    )
    search_service.kis = mock_kis_workflow

    turn_res = service.turn(opened.session_id, turn_req)
    assert turn_res.evidence_snapshot_id == "snap_feedback_created_123"
    assert len(snapshot_calls) == 1
    assert turn_res.latency["snapshot_ms"] == 42.5
    assert turn_res.latency["total_ms"] == 142.5

    # Test Undo restores initial_results
    undo_res = service.undo(
        opened.session_id,
        FeedbackUndoRequest(
            request_id="req_undo_1",
            expected_feedback_revision=turn_res.feedback_revision,
        ),
    )
    assert undo_res.evidence_snapshot_id == "snap_test"
    assert len(undo_res.results) == 1
    assert undo_res.results[0].result_id == "r_init_1"

