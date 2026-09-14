"""Tests for /api/v1/kis/search route and error mapping."""

import unittest
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from hcmai.api.contracts.kis import KISInput, KISRevisionSearchRequest, KISRevisionSearchResponse
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.api.routers.kis import create_kis_router
from hcmai.kis.models import KISEntity, KISEntityBinding, KISEvent, KISIntent
from hcmai.orchestration.errors import RevisionConflictError
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


def _make_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        inputs=["A woman cooks in kitchen."],
        language="en",
        query_text="A woman cooks in kitchen.",
        entities=[KISEntity(id="X1", kind="person", description="woman")],
        events=[
            KISEvent(
                id="E1",
                text="A woman cooks in kitchen",
                bindings=[KISEntityBinding(entity_id="X1", role="actor")],
            )
        ],
        temporal_edges=[],
    )


def _make_response() -> KISRevisionSearchResponse:
    return KISRevisionSearchResponse(
        intent=_make_intent(),
        dense_events=["A woman cooks in kitchen"],
        bm25_events=["A woman cooks in kitchen"],
        use_dense=True,
        use_bm25=True,
        results=[
            SearchResult(
                frame_id="v1_f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.9,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(query_ms=2.0, retrieval_ms=10.0),
    )


@pytest.mark.anyio
async def test_search_kis_revision_success() -> None:
    service = Mock()
    expected = _make_response()
    service.search_kis_revision.return_value = expected

    container = {"service": service}
    app = FastAPI()
    app.include_router(create_kis_router(container))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/kis/search",
            json={
                "inputs": [{"text": "A woman cooks in kitchen."}],
                "expected_revision": 0,
                "use_dense": True,
                "use_bm25": True,
                "top_k": 10,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"]["query_text"] == "A woman cooks in kitchen."
    assert data["dense_events"] == ["A woman cooks in kitchen"]
    assert len(data["results"]) == 1


@pytest.mark.anyio
async def test_search_kis_revision_dres_logging_preserves_canonical_query_text() -> None:
    service = Mock()
    expected = _make_response()
    service.search_kis_revision.return_value = expected

    vbs_mock = AsyncMock()
    vbs_mock.session_status = Mock(return_value={"connected": True})
    vbs_mock.resolve_evaluation.return_value = "eval_42"
    vbs_mock.media_item_name = Mock(return_value="video_item_1")

    container = {"service": service, "vbs_service": vbs_mock}
    app = FastAPI()
    app.include_router(create_kis_router(container))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/kis/search",
            headers={"X-VBS-User-ID": "user-test"},
            json={
                "inputs": [{"text": "A woman cooks in kitchen."}],
                "expected_revision": 0,
                "use_dense": True,
                "use_bm25": True,
                "top_k": 10,
            },
        )

    assert resp.status_code == 200
    assert resp.headers.get("X-DRES-Log-Status") == "sent"
    vbs_mock.log_results.assert_called_once()
    payload = vbs_mock.log_results.call_args[0][2]
    assert payload.events[0].value == "A woman cooks in kitchen."


@pytest.mark.anyio
async def test_search_kis_revision_error_mapping() -> None:
    service = Mock()
    container = {"service": service}
    app = FastAPI()
    app.include_router(create_kis_router(container))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 409 Conflict
        service.search_kis_revision.side_effect = RevisionConflictError("revision conflict")
        resp = await client.post(
            "/api/v1/kis/search",
            json={"inputs": [{"text": "Clue"}], "expected_revision": 0},
        )
        assert resp.status_code == 409

        # 422 Unprocessable Entity
        service.search_kis_revision.side_effect = ValueError("invalid query text")
        resp = await client.post(
            "/api/v1/kis/search",
            json={"inputs": [{"text": "Clue"}], "expected_revision": 0},
        )
        assert resp.status_code == 422

        # 503 Unavailable
        service.search_kis_revision.side_effect = SearchServiceUnavailableError("resolver down")
        resp = await client.post(
            "/api/v1/kis/search",
            json={"inputs": [{"text": "Clue"}], "expected_revision": 0},
        )
        assert resp.status_code == 503

        # 502 Bad Gateway (remote provider network error)
        service.search_kis_revision.side_effect = RuntimeError("network connection failed")
        resp = await client.post(
            "/api/v1/kis/search",
            json={"inputs": [{"text": "Clue"}], "expected_revision": 0},
        )
        assert resp.status_code == 502
