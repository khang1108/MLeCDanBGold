"""Tests for independent Search and Filter health reporting."""

from __future__ import annotations

import asyncio

import httpx
import pytest

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


class _Filter:
    """Return safe catalog health facts."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def health(self):
        """Return the standalone capability state."""

        return {
            "ready": self.ready,
            "catalog_version": "filter-v1" if self.ready else None,
            "frame_count": 470_000 if self.ready else 0,
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


@pytest.mark.parametrize("search_ready", [False, True])
@pytest.mark.parametrize("filter_ready", [False, True])
def test_filter_health_never_recalculates_search_readiness(
    search_ready: bool,
    filter_ready: bool,
) -> None:
    """Keep KIS/TRAKE availability independent from the optional catalog."""

    payload = _health(
        {
            "service": _Search(search_ready),
            "filter_service": _Filter(filter_ready) if filter_ready else None,
            "startup_messages": [],
        }
    )

    assert payload["ready"] is search_ready
    assert payload["capabilities"]["search"] is search_ready
    assert payload["capabilities"]["kis"] is search_ready
    assert payload["capabilities"]["trake"] is search_ready
    assert payload["capabilities"]["filter"] is filter_ready
    assert payload["filter_catalog"]["ready"] is filter_ready


def test_health_reports_missing_filter_without_paths() -> None:
    """Expose a bounded degraded shape without catalog internals."""

    payload = _health(
        {
            "service": _Search(True),
            "filter_service": None,
            "startup_messages": ["Filter catalog unavailable"],
        }
    )

    assert payload["filter_catalog"] == {
        "ready": False,
        "catalog_version": None,
        "frame_count": 0,
    }
    assert "path" not in str(payload["filter_catalog"]).lower()

