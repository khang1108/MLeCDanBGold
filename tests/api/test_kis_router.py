"""Tests for /api/v1/kis/search route and error mapping."""

import unittest
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from hcmai.api.contracts.kis import (
    EventPatch,
    KISOperationSummary,
    KISSearchRequest,
    KISSearchResponse,
)
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.api.routers.kis import create_kis_router
from hcmai.inference.errors import InferenceResponseError, InferenceUnavailableError
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
)
from hcmai.kis.resolver import KISResolutionError
from hcmai.orchestration.utils.errors import InvalidQueryInputError, RevisionConflictError
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.vbs.models import ApiClientAnswer, QueryEvent, QueryResultLog, RankedAnswer


def _make_intent(query_text: str | None = "A woman cooks in kitchen.") -> KISIntent:
    if query_text:
        return KISIntent(
            revision=1,
            query_text=query_text,
            entities=[KISEntity(id="X1", kind="person", description="woman")],
            events=[
                KISEvent(
                    id="E1",
                    text=query_text,
                    bindings=[KISEntityBinding(entity_id="X1", role="actor")],
                )
            ],
            temporal_edges=[],
        )
    return KISIntent(
        revision=1,
        query_text=None,
        entities=[],
        events=[
            KISEvent(
                id="E1",
                text=None,
                images=[KISImageRef(asset_id="sha256:abc", content_type="image/png")],
                bindings=[],
            )
        ],
        temporal_edges=[],
    )


def _make_response(query_text: str | None = "A woman cooks in kitchen.") -> KISSearchResponse:
    intent = _make_intent(query_text)
    return KISSearchResponse(
        intent=intent,
        operation_summary=KISOperationSummary(
            kind="initial_resolve", affected_event_ids=["E1"]
        ),
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
async def test_search_kis_success() -> None:
    service = Mock()
    expected = _make_response()
    service.search_kis.return_value = expected

    container = {"service": service}
    app = FastAPI()
    app.include_router(create_kis_router(container))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/kis/search",
            json={
                "base_intent": None,
                "expected_revision": 0,
                "operation": {
                    "kind": "initial_resolve",
                    "text": "A woman cooks in kitchen.",
                },
                "use_dense": True,
                "use_bm25": True,
                "top_k": 10,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"]["query_text"] == "A woman cooks in kitchen."
    assert data["operation_summary"]["kind"] == "initial_resolve"
    assert "exploration_seed" not in data
    assert "dense_events" not in data
    assert "bm25_events" not in data
    assert len(data["results"]) == 1


@pytest.mark.anyio
async def test_search_kis_dres_logging_preserves_canonical_query_text() -> None:
    service = Mock()
    expected = _make_response()
    service.search_kis.return_value = expected

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
                "base_intent": None,
                "expected_revision": 0,
                "operation": {
                    "kind": "initial_resolve",
                    "text": "A woman cooks in kitchen.",
                },
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
async def test_search_kis_dres_logging_falls_back_to_image_only_label() -> None:
    service = Mock()
    expected = _make_response(query_text=None)
    service.search_kis.return_value = expected

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
                "base_intent": None,
                "expected_revision": 0,
                "operation": {
                    "kind": "initial_resolve",
                    "image_refs": [{"asset_id": "sha256:abc", "content_type": "image/png"}],
                },
            },
        )

    assert resp.status_code == 200
    assert resp.headers.get("X-DRES-Log-Status") == "sent"
    vbs_mock.log_results.assert_called_once()
    payload = vbs_mock.log_results.call_args[0][2]
    assert payload.events[0].value == "[image-only KIS]"
    assert expected.intent.query_text is None


@pytest.mark.anyio
async def test_search_kis_error_mapping() -> None:
    service = Mock()
    container = {"service": service}
    app = FastAPI()
    app.include_router(create_kis_router(container))

    valid_payload = {
        "base_intent": None,
        "expected_revision": 0,
        "operation": {"kind": "initial_resolve", "text": "Clue"},
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 409 Conflict
        service.search_kis.side_effect = RevisionConflictError("revision conflict")
        resp = await client.post("/api/v1/kis/search", json=valid_payload)
        assert resp.status_code == 409

        # 422 Unprocessable Entity for an explicitly classified user-input error
        service.search_kis.side_effect = InvalidQueryInputError("invalid query text")
        resp = await client.post("/api/v1/kis/search", json=valid_payload)
        assert resp.status_code == 422

        # A generic service ValueError is an inference failure, not request validation.
        service.search_kis.side_effect = ValueError("unexpected service value error")
        resp = await client.post("/api/v1/kis/search", json=valid_payload)
        assert resp.status_code == 502

        # A provider/schema ValidationError raised during service execution is
        # also an inference failure; request validation happens before the route.
        with pytest.raises(ValidationError) as validation_error:
            EventPatch.model_validate({})
        service.search_kis.side_effect = validation_error.value
        resp = await client.post("/api/v1/kis/search", json=valid_payload)
        assert resp.status_code == 502

        # Semantic-response contract failures are bad gateway responses.
        for error in (
            KISResolutionError("invalid semantic resolution"),
            InferenceResponseError("malformed provider response"),
        ):
            service.search_kis.side_effect = error
            resp = await client.post("/api/v1/kis/search", json=valid_payload)
            assert resp.status_code == 502

        # 503 Unavailable for provider and composed-service outages.
        for error in (
            InferenceUnavailableError("provider offline"),
            SearchServiceUnavailableError("resolver down"),
        ):
            service.search_kis.side_effect = error
            resp = await client.post("/api/v1/kis/search", json=valid_payload)
            assert resp.status_code == 503

        # 502 Bad Gateway (remote provider network error)
        service.search_kis.side_effect = RuntimeError("network connection failed")
        resp = await client.post("/api/v1/kis/search", json=valid_payload)
        assert resp.status_code == 502


@pytest.mark.anyio
async def test_search_kis_request_validation_is_422() -> None:
    """FastAPI request-shape validation remains a client error boundary."""
    service = Mock()
    app = FastAPI()
    app.include_router(create_kis_router({"service": service}))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/kis/search",
            json={"expected_revision": 0},
        )

    assert resp.status_code == 422
    service.search_kis.assert_not_called()


# ---------------------------------------------------------------------------
# KIS image asset upload / display routes
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Return minimal PNG bytes for router-level tests."""
    from io import BytesIO
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (width, height), (100, 150, 200)).save(stream, format="PNG")
    return stream.getvalue()


def _make_asset_service(tmp_path: "Path") -> Mock:  # type: ignore[name-defined]
    """Return a Mock service with a real KISImageAssetStore wired in."""
    from pathlib import Path
    from hcmai.common.config import ApiConfig
    from hcmai.kis.assets import KISImageAssetStore

    store = KISImageAssetStore(
        tmp_path,
        max_upload_bytes=1024 * 1024,
        max_pixels=10_000_000,
    )
    service = Mock()
    service.kis_image_assets = store
    service.api_config = ApiConfig()
    return service


@pytest.mark.anyio
async def test_upload_then_get_asset_round_trip(tmp_path) -> None:
    """Upload returns a valid KISImageRef; GET returns identical bytes with immutable cache."""
    from pathlib import Path

    service = _make_asset_service(tmp_path)
    app = FastAPI()
    app.include_router(create_kis_router({"service": service}))
    payload = _png_bytes()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        upload_resp = await client.post(
            "/api/v1/kis/assets/images",
            files={"file": ("test.png", payload, "image/png")},
        )
        assert upload_resp.status_code == 200, upload_resp.text
        ref_json = upload_resp.json()
        assert ref_json["asset_id"].startswith("sha256:")
        assert ref_json["content_type"] == "image/png"

        get_resp = await client.get(
            f"/api/v1/kis/assets/images/{ref_json['asset_id']}"
        )
        assert get_resp.status_code == 200
        assert get_resp.content == payload
        assert get_resp.headers["content-type"].startswith("image/png")
        assert "immutable" in get_resp.headers.get("cache-control", "")


@pytest.mark.anyio
async def test_upload_invalid_mime_returns_422(tmp_path) -> None:
    service = _make_asset_service(tmp_path)
    app = FastAPI()
    app.include_router(create_kis_router({"service": service}))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/kis/assets/images",
            files={"file": ("test.bmp", b"not-an-image", "image/bmp")},
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_get_unknown_asset_returns_404(tmp_path) -> None:
    service = _make_asset_service(tmp_path)
    app = FastAPI()
    app.include_router(create_kis_router({"service": service}))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/v1/kis/assets/images/sha256:" + "a" * 64
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_asset_routes_503_when_store_unavailable() -> None:
    """When kis_image_assets is None both routes return 503."""
    service = Mock()
    service.kis_image_assets = None
    service.api_config.image_max_upload_bytes = 10 * 1024 * 1024
    app = FastAPI()
    app.include_router(create_kis_router({"service": service}))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        upload = await client.post(
            "/api/v1/kis/assets/images",
            files={"file": ("x.png", b"x", "image/png")},
        )
        assert upload.status_code == 503

        get = await client.get(
            "/api/v1/kis/assets/images/sha256:" + "f" * 64
        )
        assert get.status_code == 503
