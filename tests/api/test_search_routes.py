"""HTTP-boundary tests for explicit KIS routing."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from hcmai.api.contracts import SearchLatency, SearchRequest, SearchResponse
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