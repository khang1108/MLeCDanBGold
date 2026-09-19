"""Tests for KIS chat feedback models, session state, revisioning, and undo."""

import pytest

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackStateResponse,
    FeedbackTurnRequest,
    FeedbackUndoRequest,
    RetrievalOverride,
)
from hcmai.kis.feedback.models import (
    FeedbackCheckpoint,
    FeedbackResolution,
    FeedbackResolveContext,
    FeedbackSession,
    RefineRetrievalAction,
)
from hcmai.kis.feedback.store import (
    FeedbackError,
    FeedbackSessionStore,
    apply_prepared_refinement,
    commit_feedback_turn,
    undo_feedback,
)
from hcmai.kis.models import KISEvent, KISIntent, KISTemporalEdge


def _create_sample_intent(revision: int = 1) -> KISIntent:
    return KISIntent(
        revision=revision,
        query_text="a man is walking a dog",
        events=[
            KISEvent(id="E1", text="a man is walking a dog"),
        ],
        temporal_edges=[],
    )


def _create_sample_session(session_id: str = "sess_1") -> FeedbackSession:
    intent = _create_sample_intent(revision=1)
    state = FeedbackStateResponse(
        session_id=session_id,
        status="applied",
        feedback_revision=1,
        intent=intent,
        retrieval_overrides={},
        results=[],
        evidence_snapshot_id="snap_1",
        trail=None,
        assistant_message="Ready for feedback",
        changed_event_ids=[],
        scope="all_videos",
        can_undo=False,
    )
    return FeedbackSession(
        session_id=session_id,
        original_query="a man is walking a dog",
        use_dense=True,
        use_bm25=True,
        top_k=20,
        feedback_revision=1,
        state=state,
        history=(),
        committed_requests={},
    )


def test_undo_content_restore_with_fresh_revision():
    """Verify that Undo restores prior state while bumping feedback revision."""
    session = _create_sample_session()
    before = session.state

    after = apply_prepared_refinement(
        session,
        event_id="E1",
        text="Two people hanging a blue banner.",
    )
    assert after.intent == before.intent
    assert after.feedback_revision == before.feedback_revision + 1
    assert "E1" in after.retrieval_overrides
    assert after.can_undo is True

    restored = undo_feedback(session)
    assert restored.retrieval_overrides == before.retrieval_overrides
    assert restored.feedback_revision > after.feedback_revision
    assert restored.intent == before.intent


def test_stale_feedback_revision_rejected():
    """Request with wrong expected_feedback_revision must be rejected with STALE_REVISION."""
    session = _create_sample_session()
    request = FeedbackTurnRequest(
        request_id="req_1",
        expected_feedback_revision=999,  # Stale
        expected_kis_revision=1,
        message="Change something",
    )
    with pytest.raises(FeedbackError) as exc_info:
        commit_feedback_turn(session, request, new_state=session.state)
    assert exc_info.value.code == "STALE_REVISION"


def test_stale_kis_revision_rejected():
    """Request with wrong expected_kis_revision must be rejected with STALE_REVISION."""
    session = _create_sample_session()
    request = FeedbackTurnRequest(
        request_id="req_1",
        expected_feedback_revision=1,
        expected_kis_revision=999,  # Stale
        message="Change something",
    )
    with pytest.raises(FeedbackError) as exc_info:
        commit_feedback_turn(session, request, new_state=session.state)
    assert exc_info.value.code == "STALE_REVISION"


def test_duplicate_request_id_idempotency_and_conflict():
    """Duplicate request_id with same payload returns committed response; different payload raises conflict."""
    session = _create_sample_session()
    request = FeedbackTurnRequest(
        request_id="req_dup_1",
        expected_feedback_revision=1,
        expected_kis_revision=1,
        message="Refine E1",
    )
    target_state = session.state.model_copy(
        update={
            "feedback_revision": 2,
            "assistant_message": "Refined",
        }
    )

    committed = commit_feedback_turn(session, request, new_state=target_state)
    assert committed.feedback_revision == 2

    # Same request_id and identical payload returns committed response
    replayed = commit_feedback_turn(session, request, new_state=target_state)
    assert replayed.feedback_revision == 2
    assert replayed.assistant_message == "Refined"

    # Same request_id with different payload raises CONFLICT
    diff_request = FeedbackTurnRequest(
        request_id="req_dup_1",
        expected_feedback_revision=1,
        expected_kis_revision=1,
        message="A completely different message",
    )
    with pytest.raises(FeedbackError) as exc_info:
        commit_feedback_turn(session, diff_request, new_state=target_state)
    assert exc_info.value.code == "REQUEST_CONFLICT"


def test_session_store_ttl_and_locking():
    """Store enforces fixed TTL expiration and per-session locked context."""
    clock_time = 1000.0

    def mock_clock():
        return clock_time

    store = FeedbackSessionStore(ttl_seconds=60, max_entries=5, clock=mock_clock)
    session = _create_sample_session("s1")
    store.put(session)

    # Retrieval works before TTL
    assert store.get("s1").session_id == "s1"

    # Locked context updates session safely
    with store.locked("s1") as slot:
        slot.session = slot.session.model_copy(update={"feedback_revision": 5})

    assert store.get("s1").feedback_revision == 5

    # Advance clock past TTL
    clock_time += 61.0
    with pytest.raises(FeedbackError) as exc_info:
        store.get("s1")
    assert exc_info.value.code == "FEEDBACK_SESSION_EXPIRED"
