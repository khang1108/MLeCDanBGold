"""Tests for KIS feedback API endpoints and HTTP state contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackOpenResponse,
    FeedbackStateResponse,
    FeedbackTurnRequest,
    FeedbackUndoRequest,
    RetrievalOverride,
)
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.api.contracts.search import SearchResultMetadata
from hcmai.app import create_app
from hcmai.event_trail.models import EvidenceSnapshot, SnapshotResult, freeze_video_scores
from hcmai.event_trail.storage import EvidenceSnapshotStore
from hcmai.kis.feedback.models import (
    ClarifyAction,
    EditIntentAction,
    FeedbackAction,
    FeedbackResolveContext,
    RefineRetrievalAction,
)
from hcmai.kis.feedback.resolver import FeedbackResolver
from hcmai.kis.feedback.service import FeedbackService
from hcmai.kis.feedback.store import FeedbackError, FeedbackSessionStore
from hcmai.kis.models import KISEvent, KISIntent, KISTemporalEdge
from hcmai.retrieval.retriever.video_scores import VideoEventScores


def _create_sample_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        query_text="A person walking a dog.",
        events=[
            KISEvent(id="E1", text="A person walking a dog."),
        ],
        temporal_edges=[],
    )


def _make_search_result(
    result_id: str,
    video_id: str,
    frame_idx: int,
    timestamp_ms: int,
    score: float = 0.9,
) -> KISSearchResult:
    return KISSearchResult(
        result_id=result_id,
        video_id=video_id,
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        score=score,
        frame_id=f"{video_id}_{frame_idx}",
        frame_ids=[f"{video_id}_{frame_idx}"],
        timestamps_ms=[timestamp_ms],
        metadata=SearchResultMetadata(),
    )


@pytest.fixture
def mock_feedback_environment():
    """Sets up a realistic FeedbackService backed by stubbed search and resolver."""
    now = datetime.now(timezone.utc)
    snapshot_store = EvidenceSnapshotStore(ttl_seconds=1800, max_entries=10)

    import numpy as np

    video = VideoEventScores(
        video_id="v_001",
        frame_ids=np.array(["f1", "f2"]),
        frame_idx=np.array([10, 20]),
        timestamps_ms=np.array([10000, 20000], dtype=np.int64),
        scores=np.array([[0.8, 0.9]], dtype=np.float32),
    )
    from hcmai.orchestration.workflows.search.temporal import DecoderConfigSnapshot

    snap = EvidenceSnapshot(
        snapshot_id="snap_123",
        kis_revision=1,
        scoring_revision="score_rev_1",
        event_ids=("E1",),
        decoder_config=DecoderConfigSnapshot(0.5, 1.0, 0.0, 0),
        results={
            "r_1": SnapshotResult("r_1", "v_001", ("f2",), 0.9)
        },
        video_evidence={"v_001": freeze_video_scores(video)},
        created_at=now,
        expires_at=now,
    )
    snapshot_store.put(snap)

    class StubSearchExecution:
        def __init__(self, results, snapshot_id="snap_123"):
            self.results = results
            self.evidence_snapshot_id = snapshot_id

    class StubSearchService:
        def __init__(self):
            self.event_trail_snapshots = snapshot_store

        def execute_search(self, intent, retrieval_plan, use_dense=True, use_bm25=True, top_k=20, exclusions=None):
            res = [
                _make_search_result("r_1", "v_001", 20, 20000, score=0.95),
                _make_search_result("r_2", "v_002", 50, 50000, score=0.85),
            ]
            return StubSearchExecution(results=res)

    search_service = StubSearchService()

    fake_resolver = Mock(spec=FeedbackResolver)
    fake_resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "A person walking a brown dog in the park"},
    )

    store = FeedbackSessionStore(ttl_seconds=1800, max_entries=50)
    service = FeedbackService(
        store=store,
        resolver=fake_resolver,
        search_service=search_service,
        event_trail_service=None,
    )

    return {
        "search_service": search_service,
        "feedback_service": service,
        "resolver": fake_resolver,
        "snapshot_store": snapshot_store,
    }


@pytest.fixture
def client(mock_feedback_environment) -> TestClient:
    search_service = mock_feedback_environment["search_service"]
    feedback_service = mock_feedback_environment["feedback_service"]
    search_service.feedback = feedback_service

    app = create_app(search_service=search_service)
    return TestClient(app)


def test_feedback_api_lifecycle(client: TestClient, mock_feedback_environment) -> None:
    intent = _create_sample_intent()

    # 1. Open feedback session
    open_payload = {
        "intent": intent.model_dump(),
        "original_query": "A person walking a dog.",
        "evidence_snapshot_id": "snap_123",
        "use_dense": True,
        "use_bm25": True,
        "top_k": 20,
    }
    open_resp = client.post("/api/v1/kis/feedback/open", json=open_payload)
    assert open_resp.status_code == 200, open_resp.text
    open_data = open_resp.json()
    assert "session_id" in open_data
    assert open_data["feedback_revision"] == 1
    session_id = open_data["session_id"]
    state = open_data["state"]
    assert state["session_id"] == session_id
    assert state["feedback_revision"] == 1
    assert state["can_undo"] is False

    # Also check alias route /kis/feedback/open works
    alias_open = client.post("/kis/feedback/open", json=open_payload)
    assert alias_open.status_code == 200

    # 2. Turn 1 (apply refinement)
    turn1_payload = {
        "request_id": "req_turn_1",
        "expected_feedback_revision": 1,
        "expected_kis_revision": 1,
        "message": "Find a brown dog in a park",
    }
    turn1_resp = client.post(f"/api/v1/kis/feedback/{session_id}/turn", json=turn1_payload)
    assert turn1_resp.status_code == 200, turn1_resp.text
    turn1_data = turn1_resp.json()
    assert turn1_data["status"] == "applied"
    assert turn1_data["feedback_revision"] == 2
    assert turn1_data["can_undo"] is True
    assert "E1" in turn1_data["retrieval_overrides"]
    assert len(turn1_data["results"]) == 2

    # Duplicate request_id returns cached response
    dup_resp = client.post(f"/api/v1/kis/feedback/{session_id}/turn", json=turn1_payload)
    assert dup_resp.status_code == 200
    assert dup_resp.json()["feedback_revision"] == 2

    # Duplicate request_id with conflicting payload returns 409
    conflict_payload = dict(turn1_payload)
    conflict_payload["message"] = "Conflicting message"
    conflict_resp = client.post(f"/api/v1/kis/feedback/{session_id}/turn", json=conflict_payload)
    assert conflict_resp.status_code == 409

    # 3. Turn 2 with clarification
    mock_feedback_environment["resolver"].resolve.return_value = ClarifyAction(
        type="clarify",
        question="Did you mean walking or running?",
    )
    turn2_payload = {
        "request_id": "req_turn_2",
        "expected_feedback_revision": 2,
        "expected_kis_revision": 1,
        "message": "Maybe running?",
    }
    turn2_resp = client.post(f"/api/v1/kis/feedback/{session_id}/turn", json=turn2_payload)
    assert turn2_resp.status_code == 200
    turn2_data = turn2_resp.json()
    assert turn2_data["status"] == "clarification"
    assert turn2_data["assistant_message"] == "Did you mean walking or running?"
    assert turn2_data["feedback_revision"] == 2  # Not incremented on clarification

    # 4. Undo Turn 1
    undo_payload = {
        "request_id": "req_undo_1",
        "expected_feedback_revision": 2,
    }
    undo_resp = client.post(f"/api/v1/kis/feedback/{session_id}/undo", json=undo_payload)
    assert undo_resp.status_code == 200, undo_resp.text
    undo_data = undo_resp.json()
    assert undo_data["feedback_revision"] == 3  # Fresh revision
    assert undo_data["retrieval_overrides"] == {}  # Restored initial state
    assert undo_data["can_undo"] is False


def test_feedback_api_stale_revision_conflict(client: TestClient) -> None:
    intent = _create_sample_intent()
    open_resp = client.post(
        "/api/v1/kis/feedback/open",
        json={
            "intent": intent.model_dump(),
            "original_query": "dog",
            "evidence_snapshot_id": "snap_123",
        },
    )
    session_id = open_resp.json()["session_id"]

    # Expected revision 99 when revision is 1
    stale_resp = client.post(
        f"/api/v1/kis/feedback/{session_id}/turn",
        json={
            "request_id": "req_stale",
            "expected_feedback_revision": 99,
            "expected_kis_revision": 1,
            "message": "hello",
        },
    )
    assert stale_resp.status_code == 409, f"Got {stale_resp.status_code}: {stale_resp.text}"
    detail = stale_resp.json()["detail"]
    assert detail["code"] in ("STALE_REVISION", "FEEDBACK_REVISION_MISMATCH")


def test_feedback_api_foreign_or_missing_snapshot(client: TestClient) -> None:
    intent = _create_sample_intent()
    resp = client.post(
        "/api/v1/kis/feedback/open",
        json={
            "intent": intent.model_dump(),
            "original_query": "dog",
            "evidence_snapshot_id": "non_existent_snapshot_id",
        },
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "SNAPSHOT_NOT_FOUND"


def test_feedback_api_session_not_found(client: TestClient) -> None:
    resp = client.post(
        "/api/v1/kis/feedback/fbs_unknown_id/turn",
        json={
            "request_id": "req_notfound",
            "expected_feedback_revision": 1,
            "expected_kis_revision": 1,
            "message": "hello",
        },
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] in ("SESSION_NOT_FOUND", "FEEDBACK_SESSION_NOT_FOUND")


def test_feedback_api_service_unavailable() -> None:
    # App with no feedback service
    app = create_app(search_service=None)
    unavail_client = TestClient(app)

    intent = _create_sample_intent()
    resp = unavail_client.post(
        "/api/v1/kis/feedback/open",
        json={
            "intent": intent.model_dump(),
            "original_query": "dog",
            "evidence_snapshot_id": "snap_123",
        },
    )
    assert resp.status_code == 503
