from __future__ import annotations
import pytest
from pydantic import ValidationError
from hcmai.common.schemas import (
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

def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=" ")

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

def test_submission_contract_preserves_competition_identity() -> None:
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
