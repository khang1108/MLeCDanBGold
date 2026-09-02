"""Tests for Search health reporting while Filter remains a route stub."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from hcmai.api.routers.system import create_system_router


class _Search:
    """Return one stable Search health document for merge assertions."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def health(self, messages=()):
        """Expose the fields whose readiness Filter must not mutate."""

        return {
            "status": "ok",
            "ready": self.ready,
            "capabilities": {
                "search": self.ready,
                "kis": self.ready,
                "trake": self.ready,
            },
            "startup_messages": list(messages),
        }


def _health(container: dict[str, object]) -> dict[str, object]:
    """Read health through ASGI without starting application services."""

    app = FastAPI()
    app.include_router(create_system_router(container))

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(send())
    assert response.status_code == 200
    return response.json()


def test_health_remains_owned_by_search_runtime() -> None:
    """Do not advertise an unfinished Filter backend capability."""

    payload = _health(
        {
            "service": _Search(True),
            "startup_messages": [],
        }
    )

    assert payload["ready"] is True
    assert payload["capabilities"]["search"] is True
    assert payload["capabilities"]["kis"] is True
    assert payload["capabilities"]["trake"] is True
    assert "filter" not in payload["capabilities"]
    assert "filter_catalog" not in payload
