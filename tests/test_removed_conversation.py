from __future__ import annotations

import importlib.util

import pytest
from pydantic import ValidationError

from hcmai.common.schemas import SearchRequest, SearchResponse, SubmissionResult
from hcmai.common.schemas.enum import TaskType


def test_conversation_task_and_packages_are_removed() -> None:
    assert "kisc" not in {task.value for task in TaskType}
    assert importlib.util.find_spec("hcmai.agents") is None
    assert importlib.util.find_spec("hcmai.api.routers.kisc") is None
    assert importlib.util.find_spec("hcmai.common.schemas.conversation") is None


@pytest.mark.parametrize("legacy_field", ["session_id", "feedback"])
def test_search_request_rejects_conversation_fields(legacy_field: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "test", legacy_field: {}})


@pytest.mark.parametrize(
    "legacy_field",
    ["session_id", "turn_id", "assistant_turn_id", "ai_message"],
)
def test_search_response_rejects_conversation_fields(legacy_field: str) -> None:
    payload = {
        "request_id": "request-1",
        "query": "test",
        "query_type": "kis",
        "top_k": 1,
        "total_results": 0,
        "latency_ms": {"total": 0},
        "results": [],
        legacy_field: "legacy",
    }
    with pytest.raises(ValidationError):
        SearchResponse.model_validate(payload)


def test_submission_contract_remains_available() -> None:
    result = SubmissionResult(
        frame_id="frame-1",
        video_id="video-1",
        frame_idx=10,
        submission_code="video-1,10",
    )
    assert result.submission_code == "video-1,10"
