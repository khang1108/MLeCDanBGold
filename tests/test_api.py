"""Smoke tests for the FastAPI application endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from hcmai.app import create_app
from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.search import SearchEngine


class MockFrameStore:
    """Mock FrameStore for testing API endpoints."""

    def __init__(self) -> None:
        self.record = FrameRecord(
            frame_id="L21_V001_00000090",
            video_id="L21_V001",
            frame_idx=90,
            timestamp_ms=3600,
            image_path="/data/keyframes/L21_V001/090.jpg",
            thumbnail_path="/data/thumbnails/L21_V001/090.jpg",
            width=1920,
            height=1080,
        )
        self._records = [self.record]

    def get(self, frame_id: str) -> FrameRecord:
        if frame_id == self.record.frame_id:
            return self.record
        raise KeyError(f"Frame ID {frame_id!r} not found")


class MockRetriever:
    """Mock Retriever for testing API search."""

    def search(self, query: str, top_k: int = 10, filters: None = None) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                frame_id="L21_V001_00000090",
                source_scores={"visual": 0.95},
                final_score=0.95,
            )
        ]


@pytest.fixture
def api_client() -> TestClient:
    """Fixture providing a TestClient configured with a mock SearchEngine."""
    store = MockFrameStore()
    retriever = MockRetriever()
    engine = SearchEngine(frame_store=store, retriever=retriever)
    app = create_app(search_engine=engine)
    with TestClient(app) as client:
        yield client


def test_health_check_endpoint(api_client: TestClient) -> None:
    """Test the GET /health endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["frame_store_loaded"] is True
    assert data["total_frames"] == 1


def test_search_endpoint(api_client: TestClient) -> None:
    """Test the POST /api/v1/search endpoint."""
    payload = {"query": "một người đang đi bộ", "top_k": 5}
    response = api_client.post("/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "một người đang đi bộ"
    assert data["total_results"] == 1
    assert data["results"][0]["frame_id"] == "L21_V001_00000090"
    assert data["results"][0]["video_id"] == "L21_V001"


def test_get_frame_endpoint(api_client: TestClient) -> None:
    """Test the GET /api/v1/frames/{frame_id} endpoint."""
    response = api_client.get("/api/v1/frames/L21_V001_00000090")
    assert response.status_code == 200
    data = response.json()
    assert data["frame_id"] == "L21_V001_00000090"

    notFoundResponse = api_client.get("/api/v1/frames/UNKNOWN_FRAME")
    assert notFoundResponse.status_code == 404
