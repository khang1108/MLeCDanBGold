"""Tests for FastAPI retrieval serving application."""

from unittest.mock import Mock
from fastapi.testclient import TestClient
import pytest

from hcmai.api.contracts import ImageSearchResponse, SearchLatency, SearchResult, SearchResultMetadata
from hcmai.orchestration.workflows.search.temporal import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.serving.runtime import RetrievalRuntime
from hcmai.retrieval.serving.server import create_app
from hcmai.temporal.dp import AlignedPath
from tests.retrieval.serving.fakes import (
    make_fake_corpus,
    make_fake_temporal_search_service,
)


def _make_test_runtime() -> RetrievalRuntime:
    corpus = make_fake_corpus()
    temporal = make_fake_temporal_search_service()

    image_search = Mock()
    image_search.search.return_value = ImageSearchResponse(
        results=[
            SearchResult(
                video_id="video-1",
                frame_id="v1_f1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.91,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(query_ms=5.0, retrieval_ms=10.0, materialization_ms=1.0, total_ms=16.0),
    )

    return RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=None,
        image_search=image_search,
        active_modalities=("visual", "context"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=25_000_000,
        scoring_revision="rev-test-456",
    )


def test_health_and_capabilities_endpoints() -> None:
    app_unready = create_app(None, startup_messages=["Loading failed"])
    with TestClient(app_unready) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "NOT_SERVING"

        cap_resp = client.get("/capabilities")
        assert cap_resp.status_code == 200
        assert cap_resp.json()["ready"] is False
        assert "Loading failed" in cap_resp.json()["startup_messages"]

    runtime = _make_test_runtime()
    app_ready = create_app(runtime)
    with TestClient(app_ready) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "SERVING"

        cap_resp = client.get("/capabilities")
        assert cap_resp.status_code == 200
        data = cap_resp.json()
        assert data["ready"] is True
        assert data["scoring_revision"] == "rev-test-456"
        assert "visual" in data["active_modalities"]


def test_search_plan_endpoint() -> None:
    runtime = _make_test_runtime()
    app = create_app(runtime)
    with TestClient(app) as client:
        req = {
            "events": [
                {
                    "event_id": "E1",
                    "canonical_text": "a dog barking",
                    "dense_text": "dog bark",
                    "bm25_text": None,
                    "image_asset_ids": [],
                }
            ],
            "use_dense": True,
            "use_bm25": False,
            "top_k": 10,
        }
        resp = client.post("/search_plan", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        assert len(data["result"]["paths"]) == 2
        assert data["result"]["paths"][0]["video_id"] == "video-1"
        assert len(data["video_scores"]) == 2


def test_search_events_endpoint() -> None:
    runtime = _make_test_runtime()
    app = create_app(runtime)
    with TestClient(app) as client:
        req = {
            "original_events": ["event 1", "event 2"],
            "top_k": 10,
        }
        resp = client.post("/search_events", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["paths"]) == 2


def test_score_video_endpoint() -> None:
    runtime = _make_test_runtime()
    app = create_app(runtime)
    with TestClient(app) as client:
        req = {
            "plan": {
                "events": [{"event_id": "E1", "canonical_text": "cat", "dense_text": "cat", "bm25_text": None, "image_asset_ids": []}],
                "use_dense": True,
                "use_bm25": False,
                "top_k": 10,
            },
            "video_id": "video-1",
            "use_dense": True,
            "use_bm25": False,
        }
        resp = client.post("/score_video", json=req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["video"]["video_id"] == "video-1"


def test_score_video_not_found() -> None:
    runtime = _make_test_runtime()
    app = create_app(runtime)
    with TestClient(app) as client:
        req = {
            "plan": {
                "events": [{"event_id": "E1", "canonical_text": "cat", "dense_text": "cat", "bm25_text": None, "image_asset_ids": []}],
                "use_dense": True,
                "use_bm25": False,
                "top_k": 10,
            },
            "video_id": "unknown-video",
            "use_dense": True,
            "use_bm25": False,
        }
        resp = client.post("/score_video", json=req)
        assert resp.status_code == 500 or resp.status_code == 404


def test_search_image_endpoint() -> None:
    runtime = _make_test_runtime()
    app = create_app(runtime)
    with TestClient(app) as client:
        payload = b"\xff\xd8\xff\xe0" + b"\x00" * 32  # minimal fake JPEG header
        resp = client.post(
            "/search_image?top_k=5",
            content=payload,
            headers={"content-type": "image/jpeg"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["video_id"] == "video-1"
        assert data["candidates"][0]["frame_id"] == "v1_f1"
