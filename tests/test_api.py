"""Smoke tests for the FastAPI application endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from hcmai.app import create_app
from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.enum import RetrievalSource
from hcmai.common.schemas.retrieval import RetrievalCandidate
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class MockFrameStore:
    """Mock FrameStore for testing API endpoints."""

    def __init__(self, evidence=None) -> None:
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
        self.evidence = evidence or {}

    def get(self, frame_id: str) -> FrameRecord:
        if frame_id == self.record.frame_id:
            return self.record
        raise KeyError(f"Frame ID {frame_id!r} not found")

    get_frame = get

    @property
    def record_count(self) -> int:
        return len(self._records)

    def has_evidence(self, source) -> bool:
        return source in self.evidence

    def get_evidence(self, frame_id, source):
        store = self.evidence.get(source)
        if store is None:
            return None
        return store.get_text(frame_id)


class MockRetriever:
    """Mock Retriever for testing API search."""

    last_query_encoding_ms = 0.0
    last_index_search_ms = 0.0

    def search(
        self, query: str, top_k: int = 10, filters=None, query_type=None
    ) -> list[RetrievalCandidate]:
        return [
            RetrievalCandidate(
                frame_id="L21_V001_00000090",
                source_scores={RetrievalSource.VISUAL: 0.95},
                final_score=0.95,
            )
        ]


class MockEvidenceStore:
    """Return one fixed text value for any known frame."""

    def __init__(self, text: str) -> None:
        self.text = text

    def get_text(self, frame_id: str) -> str:
        return self.text


@pytest.fixture
def api_app() -> FastAPI:
    """Provide an ASGI app configured with a mock SearchService."""
    store = MockFrameStore()
    retriever = MockRetriever()
    service = SearchService(
        data=cast(DataService, store),
        retrieval=cast(RetrievalService, retriever),
    )
    return create_app(search_service=service)


def request(app: FastAPI, method: str, path: str, **kwargs) -> httpx.Response:
    """Send one request through the ASGI boundary without a live server."""
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()


def test_health_check_endpoint(api_app: FastAPI) -> None:
    """Test the GET /health endpoint."""
    response = request(api_app, "GET", "/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["frame_store_loaded"] is True
    assert data["total_frames"] == 1
    assert data["capabilities"]["query_types"] == {
        "kis": True,
        "vkis": True,
        "vqa": True,
        "trake": True,
    }


def test_search_materializes_configured_text_evidence() -> None:
    stores = {
        RetrievalSource.CAPTION: MockEvidenceStore("A person cooking."),
        RetrievalSource.OCR: MockEvidenceStore("BƠ"),
        RetrievalSource.ASR: MockEvidenceStore("Cho bơ vào chảo."),
    }
    service = SearchService(
        cast(DataService, MockFrameStore(stores)),
        cast(RetrievalService, MockRetriever()),
    )
    app = create_app(search_service=service)

    health = request(app, "GET", "/health").json()
    result = request(
        app, "POST", "/api/v1/search", json={"query": "cooking"}
    ).json()["results"][0]

    assert health["evidence_stores"] == {
        "caption": True, "ocr": True, "asr": True
    }
    assert (result["caption"], result["ocr_text"], result["asr_text"]) == (
        "A person cooking.", "BƠ", "Cho bơ vào chảo."
    )


def test_search_endpoint(api_app: FastAPI) -> None:
    """Test the POST /api/v1/search endpoint."""
    payload = {
        "query": "một người đang đi bộ",
        "query_type": "vkis",
        "top_k": 5,
    }
    response = request(api_app, "POST", "/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "một người đang đi bộ"
    assert data["query_type"] == "vkis"
    assert data["total_results"] == 1
    assert data["results"][0]["frame_ids"] == ["L21_V001_00000090"]
    assert data["results"][0]["video_id"] == "L21_V001"
    assert data["results"][0]["scores"]["final"] >= 0.95


@pytest.mark.parametrize(
    ("query_type", "expected_status", "expected_detail"),
    [
        ("vqa", 422, "must use /api/v1/vqa"),
    ],
)
def test_search_endpoint_routes_or_rejects_each_task_type(
    api_app: FastAPI,
    query_type: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    response = request(
        api_app,
        "POST",
        "/api/v1/search",
        json={"query": "test", "query_type": query_type},
    )
    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_degraded_service_preserves_unavailable_statuses() -> None:
    app = create_app(SearchService(data=None, retrieval=None))

    search = request(
        app, "POST", "/api/v1/search", json={"query": "red bus"}
    )
    frame = request(app, "GET", "/api/v1/frames/frame-1")
    vqa = request(
        app,
        "POST",
        "/api/v1/search",
        json={"query": "what is shown?", "query_type": "vqa"},
    )

    assert search.status_code == 503
    assert frame.status_code == 503
    assert vqa.status_code == 422


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
        "search started", "materialization started",
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


def test_missing_required_config_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The competition pipeline never substitutes defaults for missing config."""
    monkeypatch.setenv("HCMAI_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("HCMAI_METADATA_PATH", str(tmp_path / "missing.parquet"))
    monkeypatch.setenv("HCMAI_INDEX_PATH", str(tmp_path / "missing-index"))
    app = create_app()

    async def inspect_health() -> dict:
        async with app.router.lifespan_context(app):
            route = cast(APIRoute, next(
                route for route in app.routes
                if getattr(route, "path", None) == "/health"
            ))
            return await route.endpoint()

    with pytest.raises(FileNotFoundError, match="Config not found"):
        asyncio.run(inspect_health())
