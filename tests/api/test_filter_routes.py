"""HTTP tests for the literal Filter route."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from hcmai.api.contracts import FilterRequest, FilterResponse
from hcmai.api.routers.search import create_search_router
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


class _Service:
    """Record Filter delegation without loading a corpus or model."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.request = None

    def filter_frames(self, request: FilterRequest) -> FilterResponse:
        """Return one empty page or the runtime-unavailable signal."""

        if not self.available:
            raise SearchServiceUnavailableError("Literal text sources are unavailable")
        self.request = request
        return FilterResponse(
            page_id=request.page_id,
            frames_per_pages=request.frames_per_pages,
            total_pages=0,
            total_results=0,
            available_sources=["caption"],
        )


def _post(service: _Service, payload: dict[str, object]) -> httpx.Response:
    """Post through ASGI while keeping the router thread pool inline in tests."""

    app = FastAPI()
    app.include_router(create_search_router({"service": service}))

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post("/api/v1/filter", json=payload)

    return asyncio.run(send())


def test_filter_route_uses_the_new_literal_contract() -> None:
    """Delegate the keyword, free scopes, and pagination unchanged."""

    service = _Service()
    response = _post(service, {
        "query": "ao do",
        "folder_id": "L21",
        "video_id": "L21_V001",
        "frames_per_pages": 12,
        "page_id": 2,
    })

    assert response.status_code == 200
    assert service.request == FilterRequest(
        query="ao do",
        folder_id="L21",
        video_id="L21_V001",
        frames_per_pages=12,
        page_id=2,
    )
    assert response.json()["available_sources"] == ["caption"]


def test_filter_route_returns_503_without_text_sources() -> None:
    """Keep degraded startup explicit when no literal projection can be built."""

    response = _post(_Service(available=False), {"query": "hello"})

    assert response.status_code == 503
