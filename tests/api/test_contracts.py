"""Tests for the thin KIS and TRAKE HTTP contracts.

These tests lock the new API-boundary models under ``hcmai.api.contracts``.
They intentionally reject legacy task-discriminator and filter fields that
still exist in older shared schema modules during the migration window.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcmai.api.contracts import (
    SearchLatency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchResultMetadata,
    TRAKEPath,
    TRAKERequest,
    TRAKEResponse,
)


def _kis_result(**updates) -> SearchResult:
    payload = {
        "frame_id": "frame-20",
        "video_id": "video-1",
        "frame_idx": 20,
        "timestamp_ms": 2_000,
        "score": 1.5,
        "frame_ids": ["frame-10", "frame-20"],
        "timestamps_ms": [1_000, 2_000],
        "thumbnail_urls": [
            "/api/v1/frames/frame-10/thumbnail",
            "/api/v1/frames/frame-20/thumbnail",
        ],
        "frame_url": "/api/v1/frames/frame-20/image",
        "thumbnail_url": "/api/v1/frames/frame-20/thumbnail",
        "metadata": {
            "title": "Video 1",
            "caption": "chef adds butter",
            "ocr": "BUTTER",
            "objects": ["chef", "pan"],
            "asr": "add butter to the pan",
        },
    }
    payload.update(updates)
    return SearchResult.model_validate(payload)


def _trake_path(**updates) -> TRAKEPath:
    payload = {
        "video_id": "video-1",
        "score": 2.3,
        "frame_ids": ["frame-10", "frame-20"],
        "frame_idxs": [10, 20],
        "timestamps_ms": [1_000, 2_000],
        "thumbnail_urls": [
            "/api/v1/frames/frame-10/thumbnail",
            "/api/v1/frames/frame-20/thumbnail",
        ],
    }
    payload.update(updates)
    return TRAKEPath.model_validate(payload)


def test_kis_request_has_only_query_and_top_k() -> None:
    request = SearchRequest(query="chef cooks", top_k=20)

    assert request.query == "chef cooks"
    assert request.top_k == 20

    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "search_id": "legacy"})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "filters": {"video_ids": ["v1"]}})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "x", "query_type": "kis"})


def test_trake_requires_explicit_events() -> None:
    request = TRAKERequest(events=["e1", "e2"], top_k=5)

    assert request.events == ["e1", "e2"]
    assert request.top_k == 5

    with pytest.raises(ValidationError):
        TRAKERequest.model_validate({"query": "legacy prose query"})
    with pytest.raises(ValidationError):
        TRAKERequest.model_validate(
            {"events": ["e1"], "query_type": "trake"}
        )


def test_latency_contract_uses_new_stage_names() -> None:
    latency = SearchLatency()

    assert latency.model_dump() == {
        "query_ms": 0.0,
        "retrieval_ms": 0.0,
        "alignment_ms": 0.0,
        "materialization_ms": 0.0,
        "total_ms": 0.0,
    }

    with pytest.raises(ValidationError):
        SearchLatency.model_validate({"total": 10})


def test_metadata_uses_list_factory_for_objects() -> None:
    first = SearchResultMetadata()
    second = SearchResultMetadata()

    first.objects.append("chef")

    assert first.objects == ["chef"]
    assert second.objects == []


def test_kis_result_requires_aligned_path_arrays() -> None:
    result = _kis_result()

    assert result.frame_ids == ["frame-10", "frame-20"]
    assert result.metadata.objects == ["chef", "pan"]

    with pytest.raises(ValidationError, match="alignment arrays must have equal lengths"):
        _kis_result(timestamps_ms=[1_000])


def test_kis_response_requires_every_result_to_match_event_count() -> None:
    response = SearchResponse(
        query="chef cooks. chef plates.",
        events=["chef cooks", "chef plates"],
        results=[_kis_result()],
        latency=SearchLatency(total_ms=12.5),
    )

    assert response.events == ["chef cooks", "chef plates"]

    with pytest.raises(
        ValidationError,
        match="each result must contain one aligned frame per event",
    ):
        SearchResponse(
            query="chef cooks. chef plates. chef serves.",
            events=["chef cooks", "chef plates", "chef serves"],
            results=[_kis_result()],
            latency=SearchLatency(total_ms=12.5),
        )


def test_trake_path_requires_full_alignment_arrays() -> None:
    path = _trake_path()

    assert path.frame_idxs == [10, 20]

    with pytest.raises(ValidationError, match="alignment arrays must have equal lengths"):
        _trake_path(frame_idxs=[10])


def test_trake_response_requires_every_path_to_match_event_count() -> None:
    response = TRAKEResponse(
        events=["enter kitchen", "add butter"],
        paths=[_trake_path()],
        latency=SearchLatency(total_ms=9.5),
    )

    assert response.paths[0].frame_ids == ["frame-10", "frame-20"]

    with pytest.raises(ValidationError, match="one frame per event"):
        TRAKEResponse(
            events=["enter kitchen", "add butter", "plate food"],
            paths=[_trake_path()],
            latency=SearchLatency(total_ms=9.5),
        )
