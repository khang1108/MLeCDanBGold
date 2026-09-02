"""HTTP tests for stateless query-candidate generation."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
from hcmai.app import create_app
from hcmai.orchestration.pipeline import SearchService
from hcmai.query_preparation.models import QueryCandidate, QueryCandidateSet


class RecordingPreparationService:
    """Capture resolved events and return a complete candidate set."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def generate_candidates(self, events: Sequence[str]) -> QueryCandidateSet:
        """Return exactly five aligned candidates for the recorded events."""

        original = tuple(events)
        self.calls.append(original)
        return QueryCandidateSet(
            original_events=original,
            literal_en=tuple(f"literal {event}" for event in original),
            candidates=tuple(
                QueryCandidate(
                    index=index,
                    events=tuple(f"candidate {index} {event}" for event in original),
                )
                for index in range(1, 6)
        ),)


async def _request(service: SearchService, payload: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/v1/query-candidates", json=payload)


def _post(service: SearchService, payload: dict[str, Any]) -> httpx.Response:
    return asyncio.run(_request(service, payload))


def _service() -> tuple[SearchService, RecordingPreparationService]:
    preparation = RecordingPreparationService()
    service = SearchService(
        corpus=None,
        retrieval=None,
        query_preparation=preparation,  # type: ignore[arg-type]
    )
    return service, preparation


def test_kis_query_uses_deterministic_event_splitter() -> None:
    """Split raw KIS text before generation and return five candidates."""

    service, preparation = _service()

    response = _post(service, {"query": "mot. hai."})

    assert response.status_code == 200
    assert preparation.calls == [("mot", "hai")]
    assert len(response.json()["candidates"]) == 5


def test_explicit_trake_events_are_not_split_again() -> None:
    """Preserve caller-owned event boundaries exactly."""

    service, preparation = _service()

    response = _post(service, {"events": ["a. b", "c"]})

    assert response.status_code == 200
    assert preparation.calls == [("a. b", "c")]
    assert response.json()["original_events"] == ["a. b", "c"]


def test_request_requires_exactly_one_input_form() -> None:
    """Reject both and neither query input forms at the HTTP boundary."""

    service, _ = _service()

    neither = _post(service, {})
    both = _post(service, {"query": "one", "events": ["one"]})

    assert neither.status_code == 422
    assert both.status_code == 422


def test_unavailable_query_preparation_returns_503() -> None:
    """Keep ordinary SearchService construction valid without Qwen."""

    service = SearchService(corpus=None, retrieval=None)

    response = _post(service, {"query": "one"})

    assert response.status_code == 503