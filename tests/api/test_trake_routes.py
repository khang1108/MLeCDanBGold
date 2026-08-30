"""HTTP-boundary tests for explicit TRAKE routing."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from hcmai.api.contracts import SearchLatency, TRAKERequest, TRAKEResponse
from hcmai.api.routers.trake import create_trake_router
from hcmai.orchestration.pipeline import SearchService

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class _Service:
    """Record explicit TRAKE calls and return the frozen public response shape."""

    def __init__(self) -> None:
        self.requests: list[TRAKERequest] = []

    def search_trake(self, request: TRAKERequest) -> TRAKEResponse:
        """Return a successful empty TRAKE result for the supplied request."""

        self.requests.append(request)
        return TRAKEResponse(
            events=request.events,
            paths=[],
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
            return await client.post("/api/v1/trake", json=payload)

    return asyncio.run(send())


def test_trake_route_calls_explicit_trake_method() -> None:
    """Expose only the frozen TRAKE payload through the explicit service method."""

    service = _Service()
    app = FastAPI()
    app.include_router(create_trake_router({"service": service}))

    response = _post(app, {"events": ["e1", "e2"], "top_k": 3})

    assert response.status_code == 200
    assert service.requests == [TRAKERequest(events=["e1", "e2"], top_k=3)]
    assert "paths" in response.json()
    assert "submissions" not in response.json()


def test_trake_route_keeps_pydantic_validation() -> None:
    """Reject removed generic-dispatch fields at the HTTP boundary."""

    app = FastAPI()
    app.include_router(create_trake_router({"service": _Service()}))

    response = _post(
        app,
        {"events": ["e1", "e2"], "top_k": 3, "query_type": "trake"},
    )

    assert response.status_code == 422


def test_trake_route_reports_missing_service() -> None:
    """Keep degraded startup visible as an HTTP 503 response."""

    app = FastAPI()
    app.include_router(create_trake_router({"service": None}))

    response = _post(app, {"events": ["e1", "e2"], "top_k": 3})

    assert response.status_code == 503


def test_trake_route_reports_missing_runtime_dependencies() -> None:
    """Map degraded explicit TRAKE wiring to HTTP 503."""

    app = FastAPI()
    app.include_router(
        create_trake_router(
            {"service": SearchService(corpus=None, retrieval=None)}
        )
    )

    response = _post(app, {"events": ["e1", "e2"], "top_k": 3})

    assert response.status_code == 503
    assert "dependencies not loaded" in response.json()["detail"]
