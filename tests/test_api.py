"""Smoke tests for the FastAPI application endpoints."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from hcmai.app import create_app
from hcmai.retrieval.models import RetrievalSource
from hcmai.corpus import Corpus, Frame
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class MockFrameStore:
    """Mock FrameStore for testing API endpoints."""

    video_metadata_store = None

    def __init__(self, evidence=None) -> None:
        self.record = Frame(
            frame_id="L21_V001_00000090",
            video_id="L21_V001",
            frame_idx=90,
            timestamp_ms=3600,
            image_path="/data/keyframes/L21_V001/090.jpg",
            thumbnail_path="/data/thumbnails/L21_V001/090.jpg",
            fps=29.97,
        )
        self._records = [self.record]
        self.evidence = evidence or {}

    def get(self, frame_id: str) -> Frame:
        if frame_id == self.record.frame_id:
            return self.record
        raise KeyError(f"Frame ID {frame_id!r} not found")

    frame = get

    def __len__(self) -> int:
        return len(self._records)

    def frame_asset_status(self) -> SimpleNamespace:
        """Report that this in-memory fixture has no local asset resolver."""

        return SimpleNamespace(
            as_dict=lambda: {
                "ready": False,
                "checked": 0,
                "available": 0,
                "missing": 0,
            }
        )

    def has_evidence(self, source: RetrievalSource) -> bool:
        """Mirror Corpus evidence availability for health responses."""

        return source in self.evidence

    def _evidence(self, frame_id, source):
        store = self.evidence.get(source)
        if store is None:
            return None
        return store.get_text(frame_id)

    def caption(self, frame_id):
        return self._evidence(frame_id, RetrievalSource.CAPTION)

    def ocr(self, frame_id):
        return self._evidence(frame_id, RetrievalSource.OCR)

    def objects(self, frame_id):
        del frame_id
        return ()

    def title(self, video_id):
        del video_id
        return None

    def transcript(self, video_id, start_ms, end_ms):
        del video_id, start_ms, end_ms
        return self._evidence(self.record.frame_id, RetrievalSource.ASR)


class MockRetriever:
    """Mock Retriever for testing API search."""

    def score_event_videos(self, events, filters=None, **kwargs):
        """Return the one canonical visual score column used by API tests."""

        del filters, kwargs
        return [
            VideoEventScores(
                video_id="L21_V001",
                frame_ids=np.array(["L21_V001_00000090"], dtype=object),
                frame_idx=np.array([90]),
                timestamps_ms=np.array([3_600]),
                scores=np.full((len(events), 1), 0.95),
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
        corpus=cast(Corpus, store),
        retrieval=cast(RetrievalService, retriever),
    )
    return create_app(search_service=service)


def request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    raise_app_exceptions: bool = True,
    **kwargs,
) -> httpx.Response:
    """Send one request through the ASGI boundary without a live server."""
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=raise_app_exceptions
        )
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


def test_unexpected_errors_keep_cors_headers(api_app: FastAPI) -> None:
    """Expose a JSON 500 instead of masking backend failures as CORS errors."""

    async def fail_request() -> None:
        raise RuntimeError("synthetic backend failure")

    api_app.add_api_route("/test/unhandled-error", fail_request, methods=["GET"])
    response = request(
        api_app,
        "GET",
        "/test/unhandled-error",
        headers={"Origin": "http://localhost:3000"},
        raise_app_exceptions=False,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:3000"
    )


def test_cors_preflight_allows_local_frontend(api_app: FastAPI) -> None:
    """Allow the CRA development origin to preflight JSON search requests."""

    response = request(
        api_app,
        "OPTIONS",
        "/api/v1/search",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:3000"
    )


def test_health_check_endpoint(api_app: FastAPI) -> None:
    """Test the GET /health endpoint."""
    response = request(api_app, "GET", "/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["frame_store_loaded"] is True
    assert data["total_frames"] == 1
    assert data["capabilities"]["kis"] is True
    assert data["capabilities"]["trake"] is True
    assert "query_types" not in data["capabilities"]


def test_uninitialized_health_exposes_only_kis_and_trake() -> None:
    app = create_app()

    response = request(app, "GET", "/health")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["search"] is False
    assert capabilities["kis"] is False
    assert capabilities["trake"] is False
    assert "query_types" not in capabilities
    assert "vqa" not in capabilities
    assert "vkis" not in capabilities


def test_search_materializes_configured_text_evidence() -> None:
    stores = {
        RetrievalSource.CAPTION: MockEvidenceStore("A person cooking."),
        RetrievalSource.OCR: MockEvidenceStore("BƠ"),
        RetrievalSource.ASR: MockEvidenceStore("Cho bơ vào chảo."),
    }
    service = SearchService(
        cast(Corpus, MockFrameStore(stores)),
        cast(RetrievalService, MockRetriever()),
    )
    app = create_app(search_service=service)

    health = request(app, "GET", "/health").json()
    result = request(
        app, "POST", "/api/v1/search", json={"query": "cooking"}
    ).json()["results"][0]

    metadata = result["metadata"]
    assert (metadata["caption"], metadata["ocr"], metadata["asr"]) == (
        "A person cooking.", "BƠ", "Cho bơ vào chảo."
    )


def test_search_endpoint(api_app: FastAPI) -> None:
    """Test the POST /api/v1/search endpoint."""
    payload = {
        "query": "một người đang đi bộ",
        "top_k": 5,
    }
    response = request(api_app, "POST", "/api/v1/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "một người đang đi bộ"
    assert data["events"] == ["một người đang đi bộ"]
    assert len(data["results"]) == 1
    assert set(data["latency"]) == {
        "query_ms",
        "retrieval_ms",
        "alignment_ms",
        "materialization_ms",
        "total_ms",
    }
    assert data["results"][0]["frame_ids"] == ["L21_V001_00000090"]
    assert data["results"][0]["video_id"] == "L21_V001"
    assert data["results"][0]["score"] == pytest.approx(0.95)
    assert data["results"][0]["fps"] == 29.97


def test_vqa_payload_is_rejected_by_kis_search_schema(api_app: FastAPI) -> None:
    response = request(
        api_app,
        "POST",
        "/api/v1/search",
        json={"query": "test", "query_type": "vqa"},
    )
    assert response.status_code == 422
    assert "must use /api/v1/vqa" not in str(response.json())


def test_vqa_route_is_not_registered(api_app: FastAPI) -> None:
    response = request(
        api_app,
        "POST",
        "/api/v1/vqa",
        json={"query": "what is shown?"},
    )
    assert response.status_code == 404


def test_degraded_service_preserves_unavailable_statuses() -> None:
    app = create_app(SearchService(corpus=None, retrieval=None))

    search = request(
        app, "POST", "/api/v1/search", json={"query": "red bus"}
    )
    frame = request(app, "GET", "/api/v1/frames/frame-1")

    assert search.status_code == 503
    assert frame.status_code == 503


def test_search_endpoint_exposes_frozen_latency_stages(api_app: FastAPI) -> None:
    """Expose only the five public Phase A timing stages."""

    response = request(
        api_app,
        "POST",
        "/api/v1/search",
        json={"query": "red bus", "top_k": 5},
    )

    assert response.status_code == 200
    assert set(response.json()["latency"]) == {
        "query_ms",
        "retrieval_ms",
        "alignment_ms",
        "materialization_ms",
        "total_ms",
    }


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
