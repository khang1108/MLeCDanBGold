"""Contract tests for hosted query-preparation endpoints."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
from llm.config import LLMServiceConfig
from llm.pipeline import LLMService
from llm.server.api import create_llm_app


class FakeQueryPreparationRuntime:
    """Small injectable runtime that avoids loading GPU models."""

    config = LLMServiceConfig.from_yaml("llm/config.yaml")
    translation = ["a chef holds X", "the chef rolls X"]

    @staticmethod
    def load() -> None:
        """Match the hosted runtime lifecycle without allocating resources."""

    def translate_query_events(self, events: list[str]) -> list[str]:
        """Return scripted aligned translations."""

        assert len(events) == len(self.translation)
        return list(self.translation)

    @staticmethod
    def generate_query_candidates(events: list[str], candidate_count: int = 5) -> dict[str, Any]:
        """Return a valid deterministic candidate response."""
        return {
            "literal_en": [f"literal {index}" for index, _ in enumerate(events)],
            "candidates": [
                [f"candidate {candidate}:{event}" for event, _ in enumerate(events)]
                for candidate in range(candidate_count)
        ],}



async def _request(app: Any, path: str, payload: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, json=payload)


def _post(path: str, payload: dict[str, Any]) -> httpx.Response:
    app = create_llm_app(cast(LLMService, FakeQueryPreparationRuntime()))
    return asyncio.run(_request(app, path, payload))


def test_translate_query_events_returns_one_output_per_input() -> None:
    """Expose aligned structured translations rather than model prose."""

    response = _post(
        "/query-preparation/translate",
        {"events": ["dau bep cam X", "dau bep lan X"]},
    )

    assert response.status_code == 200
    assert response.json()["events"] == FakeQueryPreparationRuntime.translation


def test_candidate_endpoint_returns_exactly_five_aligned_bundles() -> None:
    """Enforce the frozen candidate count and input event shape."""

    response = _post(
        "/query-preparation/candidates",
        {"events": ["E1", "E2"], "candidate_count": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["literal_en"]) == 2
    assert len(body["candidates"]) == 5
    assert all(len(item) == 2 for item in body["candidates"])


def test_query_preparation_rejects_empty_events_and_wrong_candidate_count() -> None:
    """Reject malformed work before invoking the model owner."""

    empty = _post("/query-preparation/translate", {"events": []})
    wrong_count = _post(
        "/query-preparation/candidates",
        {"events": ["E1"], "candidate_count": 4},
    )

    assert empty.status_code == 422
    assert wrong_count.status_code == 422
