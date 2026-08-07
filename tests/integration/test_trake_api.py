"""API integration test for the unified task contract and TRAKE pipeline.

Proves the handover boundary end to end: HTTP body -> ``TaskRequest``
discriminator -> ``PipelineRegistry`` -> ``TRAKEPipeline.execute`` ->
``TRAKEResponse`` -> HTTP 200 JSON, with fake models and a tiny fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hcmai.app import create_app
from hcmai.common.schemas import RetrievalSource
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.task_router import PipelineRegistry
from hcmai.retriever.pipeline import RetrievalService
from hcmai.retriever.video_scores import VideoEventScores

_MAPPING = pd.DataFrame(
    {
        "embedding_index": [0, 1, 2],
        "frame_id": ["frame_10", "frame_20", "frame_30"],
        "video_id": ["video_001", "video_001", "video_002"],
        "frame_idx": [10, 20, 30],
        "timestamp_ms": [1_000, 2_000, 3_000],
    }
)
# Event 1 peaks on frame_10 and event 2 on frame_20, so the only strictly
# chronological path of video_001 is (frame_10, frame_20).
_SCORES = np.array([[0.9, 0.1, 0.4], [0.2, 0.8, 0.5]], dtype=np.float32)

_FRAMES = {
    row.frame_id: SimpleNamespace(
        frame_id=row.frame_id,
        video_id=row.video_id,
        frame_idx=row.frame_idx,
        timestamp_ms=row.timestamp_ms,
    )
    for row in _MAPPING.itertuples()
}


class _FakeRetrieval:
    """Shortlist video_001 only, so video_002 never reaches the aligner."""

    def score_visual_videos(
        self,
        events: Sequence[str],
        top_k: int = 500,
        max_videos: int = 200,
        rrf_k: int = 60,
        chunk_size: int = 65_536,
    ) -> list[VideoEventScores]:
        del top_k, max_videos, rrf_k, chunk_size
        assert len(events) == len(_SCORES)
        return [
            VideoEventScores(
                video_id="video_001",
                frame_ids=np.array(["frame_10", "frame_20"], dtype=object),
                frame_idx=np.array([10, 20]),
                timestamps_ms=np.array([1_000.0, 2_000.0]),
                scores=_SCORES[:, [0, 1]],
            )
        ]


class _FakeData:
    record_count = len(_FRAMES)

    def get_frame(self, frame_id: str):
        if frame_id not in _FRAMES:
            raise KeyError(frame_id)
        return _FRAMES[frame_id]

    def get_evidence(self, frame_id: str, source: RetrievalSource) -> None:
        del frame_id, source
        return None

    def has_evidence(self, source: RetrievalSource) -> bool:
        del source
        return False


_BODY = {
    "query_type": "trake",
    "query": "enter kitchen -> add butter",
    "events": ["enter kitchen", "add butter"],
    "top_k": 20,
}


def _client(
    data: Any = None,
    retrieval: Any = None,
    registry: PipelineRegistry | None = None,
) -> TestClient:
    service = SearchService(
        cast(DataService, data),
        cast(RetrievalService, retrieval),
        pipeline_registry=registry,
    )
    return TestClient(create_app(service))


@pytest.fixture
def client() -> TestClient:
    return _client(_FakeData(), _FakeRetrieval())


def test_trake_request_reaches_the_pipeline_and_returns_a_submission(
    client: TestClient,
) -> None:
    with client:
        response = client.post("/api/v1/search", json=_BODY)

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "trake"
    assert payload["request_id"].startswith("trake-")
    assert payload["query"] == _BODY["query"]
    assert payload["events"] == _BODY["events"]
    assert payload["top_k"] == 20
    assert payload["total_results"] == 1
    assert payload["submissions"] == [
        {
            "rank": 1,
            "video_id": "video_001",
            "frame_ids": ["frame_10", "frame_20"],
            "frame_idxs": [10, 20],
            "warnings": [],
        }
    ]
    assert payload["warnings"] == []


def test_every_submission_has_one_frame_per_event(client: TestClient) -> None:
    with client:
        payload = client.post("/api/v1/search", json=_BODY).json()

    event_count = len(payload["events"])
    assert all(
        len(row["frame_ids"]) == event_count == len(row["frame_idxs"])
        for row in payload["submissions"]
    )


def test_kis_still_works_through_the_union_contract(client: TestClient) -> None:
    with client:
        response = client.post(
            "/api/v1/search",
            json={"query_type": "kis", "query": "a red bus", "top_k": 5},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "kis"
    assert payload["results"][0]["frame_id"] == "frame_10"
    assert payload["results"][0]["video_id"] == "video_001"
    assert payload["results"][0]["frame_idx"] == 10


def test_health_reports_trake_once_the_pipeline_is_registered(
    client: TestClient,
) -> None:
    with client:
        capabilities = client.get("/health").json()["capabilities"]

    assert capabilities["query_types"]["trake"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"query_type": "trake", "query": "only one event", "events": ["one"]},
        {"query_type": "trake", "top_k": 20},
        {"query_type": "trake", "query": "prose query with no events field"},
        {"query_type": "trake", "query": "enter kitchen | add butter"},
    ],
)
def test_invalid_trake_input_is_rejected(
    client: TestClient, body: dict[str, Any]
) -> None:
    with client:
        assert client.post("/api/v1/search", json=body).status_code == 422


def test_unregistered_pipeline_is_not_implemented() -> None:
    with _client(_FakeData(), _FakeRetrieval(), PipelineRegistry()) as client:
        response = client.post("/api/v1/search", json=_BODY)

    assert response.status_code == 501


def test_unavailable_dependency_is_service_unavailable() -> None:
    with _client(_FakeData(), None) as client:
        response = client.post("/api/v1/search", json=_BODY)

    assert response.status_code == 503
