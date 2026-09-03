"""HTTP-boundary tests for explicit KIS routing."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from hcmai.api.contracts import (
    ImageSearchResponse,
    SearchLatency,
    SearchRequest,
    SearchResponse,
)
from hcmai.api.routers.search import create_search_router
from hcmai.orchestration.pipeline import SearchService

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class _Service:
    """Record explicit KIS calls and return the frozen public response shape."""

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search_kis(self, request: SearchRequest) -> SearchResponse:
        """Return a successful empty KIS result for the supplied request."""

        self.requests.append(request)
        return SearchResponse(
            query=request.query,
            events=[request.query],
            dense_events=[request.query] if request.use_dense else None,
            bm25_caption_events=[request.query] if request.use_bm25 else None,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            results=[],
            latency=SearchLatency(),
        )


def _post(app: FastAPI, payload: dict[str, object]) -> httpx.Response:
    """Send one request through ASGI without starting repository services."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/v1/search", json=payload)

    return asyncio.run(send())


def _post_image(
    app: FastAPI,
    payload: bytes,
    *,
    content_type: str = "image/jpeg",
    top_k: int = 3,
) -> httpx.Response:
    """Upload one image through the public multipart route."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/v1/search/image",
                data={"top_k": str(top_k)},
                files={"image": ("query.jpg", payload, content_type)},
            )

    return asyncio.run(send())


def _jpeg() -> bytes:
    """Build one small valid upload without fixture files."""

    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, "JPEG")
    return output.getvalue()


def test_search_route_calls_explicit_kis_method() -> None:
    """Expose only the frozen KIS payload through the explicit service method."""

    service = _Service()
    app = FastAPI()
    app.include_router(create_search_router({"service": service}))

    response = _post(app, {"query": "chef cooks", "top_k": 3})

    assert response.status_code == 200
    assert service.requests == [SearchRequest(query="chef cooks", top_k=3)]
    assert "query_type" not in response.json()


def test_search_route_keeps_pydantic_validation() -> None:
    """Reject removed generic-dispatch fields at the HTTP boundary."""

    app = FastAPI()
    app.include_router(create_search_router({"service": _Service()}))

    response = _post(
        app,
        {"query": "chef cooks", "top_k": 3, "query_type": "kis"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " \n\t ", "top_k": 1},
        {"query": "chef cooks", "top_k": 0},
],)
def test_search_route_rejects_invalid_values_at_http_boundary(
    payload: dict[str, object],
) -> None:
    """Return FastAPI validation responses instead of letting workflows raise."""

    app = FastAPI()
    app.include_router(create_search_router({"service": _Service()}))

    assert _post(app, payload).status_code == 422


def test_search_route_reports_missing_service() -> None:
    """Keep degraded startup visible as an HTTP 503 response."""

    app = FastAPI()
    app.include_router(create_search_router({"service": None}))

    response = _post(app, {"query": "chef cooks", "top_k": 3})

    assert response.status_code == 503


def test_search_route_reports_missing_runtime_dependencies() -> None:
    """Map degraded explicit KIS wiring to HTTP 503."""

    app = FastAPI()
    app.include_router(
        create_search_router({"service": SearchService(corpus=None, retrieval=None)})
    )

    response = _post(app, {"query": "chef cooks", "top_k": 3})

    assert response.status_code == 503
    assert "dependencies not loaded" in response.json()["detail"]


def test_image_search_route_delegates_multipart_payload() -> None:
    """Pass one bounded image and validated top-k to the image service."""

    class ImageService:
        SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
        max_upload_bytes = 1024

    class Service:
        image_search = ImageService()

        def __init__(self) -> None:
            self.call: tuple[bytes, str | None, int] | None = None

        def search_image(
            self,
            payload: bytes,
            *,
            content_type: str | None,
            top_k: int,
        ) -> ImageSearchResponse:
            self.call = (payload, content_type, top_k)
            return ImageSearchResponse(results=[], latency=SearchLatency())

    service = Service()
    app = FastAPI()
    app.include_router(create_search_router({"service": service}))
    payload = _jpeg()

    response = _post_image(app, payload, top_k=7)

    assert response.status_code == 200
    assert service.call == (payload, "image/jpeg", 7)
    assert response.json()["results"] == []


def test_image_search_route_rejects_unsupported_media_type() -> None:
    """Reject non-image uploads before invoking model inference."""

    service = SimpleNamespace(
        image_search=SimpleNamespace(
            SUPPORTED_MEDIA_TYPES={"image/jpeg", "image/png", "image/webp"},
            max_upload_bytes=1024,
        )
    )
    app = FastAPI()
    app.include_router(create_search_router({"service": service}))

    response = _post_image(app, b"not an image", content_type="text/plain")

    assert response.status_code == 415
