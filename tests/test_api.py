"""Smoke tests for the FastAPI application endpoints."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI

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
def api_app() -> FastAPI:
    """Provide an ASGI app configured with a mock SearchEngine."""
    store = MockFrameStore()
    retriever = MockRetriever()
    engine = SearchEngine(frame_store=store, retriever=retriever)
    return create_app(search_engine=engine)


def request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    """Send one request through the ASGI boundary without a live server."""
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_check_endpoint(api_app: FastAPI) -> None:
    """Test the GET /health endpoint."""
    response = request(api_app, "GET", "/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["frame_store_loaded"] is True
    assert data["total_frames"] == 1


def test_search_endpoint(api_app: FastAPI) -> None:
    """Test the POST /api/v1/search endpoint."""
    payload = {"query": "một người đang đi bộ", "top_k": 5}
    response = request(api_app, "POST", "/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "một người đang đi bộ"
    assert data["total_results"] == 1
    assert data["results"][0]["frame_id"] == "L21_V001_00000090"
    assert data["results"][0]["video_id"] == "L21_V001"
    assert data["results"][0]["scores"]["final"] == 0.95


def test_search_endpoint_logs_every_pipeline_stage(
    api_app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """Operators can see progress before and after every online search stage."""
    with caplog.at_level(logging.INFO, logger="hcmai"):
        response = request(api_app, "POST", "/api/v1/search",
                           json={"query": "red bus", "top_k": 5})
    assert response.status_code == 200
    output = "\n".join(record.getMessage() for record in caplog.records)
    stages = (
        "search started", "retrieval started", "retrieval completed candidates=1",
        "fusion skipped", "reranking skipped", "materialization started",
        "search completed results=1",
    )
    assert all(stage in output for stage in stages)


def test_get_frame_endpoint(api_app: FastAPI) -> None:
    """Test the GET /api/v1/frames/{frame_id} endpoint."""
    response = request(api_app, "GET", "/api/v1/frames/L21_V001_00000090")
    assert response.status_code == 200
    data = response.json()
    assert data["frame_id"] == "L21_V001_00000090"

    notFoundResponse = request(api_app, "GET", "/api/v1/frames/UNKNOWN_FRAME")
    assert notFoundResponse.status_code == 404


def test_list_session_ids_endpoint(api_app: FastAPI) -> None:
    """List every in-memory conversation ID in creation order."""
    first = request(api_app, "POST", "/api/v1/session").json()["session_id"]
    second = request(api_app, "POST", "/api/v1/session").json()["session_id"]

    response = request(api_app, "GET", "/api/v1/sessions")

    assert response.status_code == 200
    assert response.json() == [first, second]


def test_missing_artifacts_do_not_prevent_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The API stays live and reports not-ready without local corpus files."""
    monkeypatch.setenv("HCMAI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("HCMAI_METADATA_PATH", str(tmp_path / "missing.parquet"))
    monkeypatch.setenv("HCMAI_INDEX_PATH", str(tmp_path / "missing-index"))
    app = create_app()

    async def inspect_health() -> dict:
        async with app.router.lifespan_context(app):
            route = next(
                route for route in app.routes
                if getattr(route, "path", None) == "/health"
            )
            return await route.endpoint()

    health = asyncio.run(inspect_health())
    assert health["status"] == "ok"
    assert health["ready"] is False
    assert health["frame_store_loaded"] is False
    assert health["retriever_loaded"] is False
    assert health["startup_messages"]
