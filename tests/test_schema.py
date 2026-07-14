"""Validation tests for the shared project contracts."""

import json

import pytest
from pydantic import ValidationError

from hcmai.schema import (
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchScores,
    SearchLatency,
    SearchFilters,
    FrameRecord,
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


@pytest.mark.parametrize("top_k", [0, 101])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="a person walking", top_k=top_k)


def test_negative_frame_idx_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FrameRecord(
            frame_id="frame-001",
            video_id="video-001",
            frame_idx=-1,
            timestamp_ms=500,
            image_path="frames/frame-001.jpg",
            width=1920,
            height=1080,
        )


def test_invalid_time_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(start_time_ms=2_000, end_time_ms=1_000)


def test_valid_request_and_response_can_be_serialized_to_json() -> None:
    request = SearchRequest(query=" a person walking ", top_k=5)
    response = make_valid_response()

    request_data = json.loads(request.model_dump_json())
    response_data = json.loads(response.model_dump_json())

    assert request_data["query"] == "a person walking"
    assert request_data["top_k"] == 5
    assert response_data["search_mode"] == "accuracte"
    assert response_data["results"][0]["scores"]["final"] == 0.95


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="a person walking", unexpected_field=True)
