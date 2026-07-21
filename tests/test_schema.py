"""Validation tests for the shared project contracts."""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from hcmai.common.schemas import (
    ConversationSession,
    FrameFeedback,
    FrameRecord,
    MessageRequest,
    MessageResponse,
    SearchFilters,
    SearchLatency,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
    SubmissionResult,
)


def make_valid_response() -> SearchResponse:
    """Build the smallest valid search response."""
    return SearchResponse(
        request_id="request-001",
        query="a person walking",
        search_mode=SearchMode.ACCURATE,
        top_k=1,
        total_results=1,
        latency_ms=SearchLatency(total=25),
        results=[
            SearchResult(
                rank=1,
                frame_id="frame-001",
                video_id="video-001",
                frame_idx=10,
                timestamp_ms=500,
                scores=SearchScores(final=0.95),
            )
        ],
    )


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


def test_search_request_and_response_kisc_fields() -> None:
    feedback = FrameFeedback(accepted_frame_ids=["f1"], rejected_frame_ids=["f2"])
    req = SearchRequest(query="test", session_id="sess-01", feedback=feedback)
    assert req.session_id == "sess-01"
    assert req.feedback.accepted_frame_ids == ["f1"]

    resp = make_valid_response()
    resp.session_id = "sess-01"
    resp.turn_id = "turn-01"
    resp.ai_message = "Found 1 frame"
    assert resp.session_id == "sess-01"


def test_message_aliases_work_identically() -> None:
    msg_req = MessageRequest(query="hello")
    assert isinstance(msg_req, SearchRequest)


def test_conversation_session_and_submission_result() -> None:
    sess = ConversationSession(session_id="s1", created_at=1000)
    assert sess.session_id == "s1"

    sub = SubmissionResult(
        frame_id="f1", video_id="L21_V001", frame_idx=10, submission_code="L21_V001,10"
    )
    assert sub.submission_code == "L21_V001,10"
