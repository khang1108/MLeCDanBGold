"""HTTP-boundary tests for the Filter development placeholder."""

from __future__ import annotations

import asyncio

import httpx

from fastapi import FastAPI

from hcmai.api.routers.filter import create_filter_router


def _post(payload: object) -> httpx.Response:
    """Send one request through ASGI without starting repository services."""

    app = FastAPI()
    app.include_router(create_filter_router())

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/v1/filter", json=payload)

    return asyncio.run(send())


def test_filter_route_reports_feature_under_development() -> None:
    """Keep the frontend route stable without exposing unfinished behavior."""

    response = _post({"metadata_filters": {"caption": "áo đỏ"}})

    assert response.status_code == 501
    assert response.json() == {
        "detail": "Tính năng Filter đang được phát triển"
    }


def test_filter_route_does_not_validate_or_execute_filter_payloads() -> None:
    """Return the same placeholder response for any temporary client contract."""

    response = _post(["unfinished", {"shape": True}])

    assert response.status_code == 501
    assert response.json()["detail"] == "Tính năng Filter đang được phát triển"
