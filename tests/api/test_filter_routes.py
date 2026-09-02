"""HTTP-boundary tests for the independent Filter endpoint."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from fastapi import FastAPI

from hcmai.api.contracts import FilterRequest, FilterResponse
from hcmai.api.routers.filter import create_filter_router
from hcmai.filtering.service import FilterServiceUnavailableError

pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


class _Service:
    """Record validated Filter calls and return an empty complete page."""

    def __init__(self) -> None:
        self.requests: list[FilterRequest] = []

    def filter(self, request: FilterRequest) -> FilterResponse:
        """Return a contract-valid response echoing pagination coordinates."""

        self.requests.append(request)
        return FilterResponse(
            page_id=request.page_id,
            frames_per_pages=request.frames_per_pages,
            total_results=0,
            total_pages=0,
            results=[],
        )


class _UnavailableService:
    """Represent a catalog that became unavailable after startup."""

    def filter(self, request: FilterRequest) -> FilterResponse:
        """Raise the service-level availability error mapped to HTTP 503."""

        raise FilterServiceUnavailableError("Filter catalog closed")


def _post(app: FastAPI, payload: dict[str, object]) -> httpx.Response:
    """Send one Filter request through ASGI without repository startup."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/v1/filter", json=payload)

    return asyncio.run(send())


def test_filter_route_passes_strict_request_to_service() -> None:
    """Keep SQL and matching out of the transport layer."""

    service = _Service()
    app = FastAPI()
    app.include_router(create_filter_router({"filter_service": service}))

    response = _post(
        app,
        {
            "metadata_filters": {"title": "Áo đỏ"},
            "folder_id": "L21",
            "frames_per_pages": 24,
            "page_id": 2,
        },
    )

    assert response.status_code == 200
    assert service.requests == [
        FilterRequest(
            metadata_filters={"title": "ao do"},
            folder_id="L21",
            frames_per_pages=24,
            page_id=2,
        )
    ]
    assert response.json()["frames_per_pages"] == 24


def test_filter_route_rejects_invalid_input_before_service_call() -> None:
    """Return FastAPI 422 for requests outside bounded pagination."""

    service = _Service()
    app = FastAPI()
    app.include_router(create_filter_router({"filter_service": service}))

    response = _post(app, {"frames_per_pages": 49})

    assert response.status_code == 422
    assert service.requests == []


@pytest.mark.parametrize(
    "container",
    [
        {"filter_service": None},
        {"filter_service": _UnavailableService()},
    ],
)
def test_filter_route_maps_only_unavailable_state_to_503(
    container: dict[str, object],
) -> None:
    """Expose optional deployment degradation without affecting Search."""

    app = FastAPI()
    app.include_router(create_filter_router(container))

    response = _post(app, {})

    assert response.status_code == 503


def test_filter_route_does_not_hide_catalog_corruption() -> None:
    """Allow unexpected invariant failures to reach application 500 handling."""

    error = RuntimeError("corrupt catalog")
    app = FastAPI()
    app.include_router(
        create_filter_router({"filter_service": None, "filter_error": error})
    )

    with pytest.raises(RuntimeError, match="corrupt catalog"):
        _post(app, {})

