from __future__ import annotations
import pytest
from pydantic import ValidationError
from hcmai.common.schemas import (
    ConversationConstraint,
    ConversationSession,
    ConversationTurn,
    FrameFeedback,
    MessageRequest,
    MessageResponse,
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
    TaskType,
)

def _response(**updates) -> SearchResponse:
    payload = {
        "request_id": "request-001",
        "query": "a person walking",
        "query_type": TaskType.KIS,
        "top_k": 1,
        "total_results": 0,
        "latency_ms": SearchLatency(total=25),
        "results": [],
    }
    payload.update(updates)
    return SearchResponse.model_validate(payload)

def test_empty_query_and_stateless_feedback_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=" ")
    with pytest.raises(ValidationError, match="feedback requires session_id"):
        SearchRequest(query="test", feedback=FrameFeedback())

def test_retired_search_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(
            {"query": "test", "search_mode": "accurate"}
        )

def test_query_type_is_typed_and_defaults_to_kis() -> None:
    assert SearchRequest(query="test").query_type is TaskType.KIS
    assert (
        SearchRequest.model_validate(
            {"query": "test", "query_type": "vkis"}
        ).query_type
        is TaskType.VKIS
    )
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(
            {"query": "test", "query_type": "unknown"}
        )

def test_feedback_is_deduplicated_and_disjoint() -> None:
    feedback = FrameFeedback(accepted_frame_ids=["f1", "f1"])
    assert feedback.accepted_frame_ids == ["f1"]
    with pytest.raises(ValidationError, match="must be disjoint"):
        FrameFeedback(
            accepted_frame_ids=["f1"],
            rejected_frame_ids=["f1"],
        )

def test_conversation_turn_role_is_typed() -> None:
    turn = ConversationTurn(
        turn_id="turn-1",
        sender="user",
        message="find the red car",
        created_at=100,
    )
    assert turn.sender == "user"
    with pytest.raises(ValidationError):
        ConversationTurn.model_validate(
            {**turn.model_dump(), "sender": "system"}
        )
    constraint = ConversationConstraint(
        slot="color",
        value="blue",
        polarity="positive",
        source_turn_id=turn.turn_id,
    )
    assert constraint.slot == "color"
    with pytest.raises(ValidationError):
        ConversationConstraint.model_validate(
            {**constraint.model_dump(), "polarity": "maybe"}
        )

def test_kisc_response_requires_complete_turn_context() -> None:
    response = _response(
        session_id="session-1",
        turn_id="turn-1",
        assistant_turn_id="turn-2",
        ai_message="Retrieved 0 frame candidates.",
    )
    assert response.assistant_turn_id == "turn-2"
    with pytest.raises(ValidationError, match="complete turn metadata"):
        _response(session_id="session-1", turn_id="turn-1")

def test_alias_session_and_submission_contracts() -> None:
    assert isinstance(MessageRequest(query="hello"), SearchRequest)
    assert MessageResponse is SearchResponse
    session = ConversationSession(
        session_id="session-1",
        created_at=100,
        problem_id="problem-7",
    )
    assert session.problem_id == "problem-7"
    valid = SubmissionResult(
        frame_id="f1",
        video_id="L21_V001",
        frame_idx=10,
        submission_code="L21_V001,10",
    )
    with pytest.raises(ValidationError, match="submission_code"):
        SubmissionResult.model_validate(
            {**valid.model_dump(), "submission_code": "L21_V001,11"}
        )
