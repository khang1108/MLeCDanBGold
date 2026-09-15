"""Successful search routes forward complete QueryResultLog events safely."""

from __future__ import annotations

import asyncio
import hashlib
import io
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from hcmai.api.contracts.filter import FilterResult, FilterResponse
from hcmai.api.contracts.kis import KISRevisionSearchResponse
from hcmai.api.contracts.latency import SearchLatency
from hcmai.api.contracts.search import (
    ImageSearchResponse,
    SearchResult,
    SearchResultMetadata,
)
from hcmai.api.routers.kis import create_kis_router
from hcmai.api.routers.search import create_search_router
from hcmai.kis.models import KISEvent, KISIntent
from hcmai.vbs.config import DresSettings


EPOCH_MS = 1_800_000_123_456


class _SearchService:
    """Return fixed canonical results for text, image, and filter routes."""

    def __init__(self) -> None:
        self.image_search = SimpleNamespace(
            SUPPORTED_MEDIA_TYPES={"image/jpeg", "image/png", "image/webp"},
            max_upload_bytes=1024 * 1024,
        )

    def search_kis_revision(self, request) -> KISRevisionSearchResponse:
        del request
        intent = KISIntent(
            revision=1,
            language="en",
            query_text="person running",
            events=[KISEvent(id="E1", text="person running")],
        )
        return KISRevisionSearchResponse(
            intent=intent,
            exploration_seed={
                "semantic_revision": 1,
                "events": [{
                    "event_id": "E1",
                    "canonical_text": "person running",
                    "dense_text": "person running",
                    "bm25_text": "person running",
                }],
                "use_dense": True,
                "use_bm25": True,
            },
            use_dense=True,
            use_bm25=True,
            results=[_search_result("video-a", 100), _search_result("video-b", 200)],
            latency=SearchLatency(),
        )

    def search_image(self, payload: bytes, *, content_type: str | None, top_k: int) -> ImageSearchResponse:
        del payload, content_type, top_k
        return ImageSearchResponse(
            results=[_search_result("video-a", 300)],
            latency=SearchLatency(),
        )

    def filter_frames(self, request) -> FilterResponse:
        return FilterResponse(
            page_id=request.page_id,
            frames_per_pages=request.frames_per_pages,
            total_pages=3,
            total_results=42,
            available_sources=["caption", "objects"],
            results=[
                FilterResult(
                    frame_id="filter-1",
                    video_id="video-a",
                    frame_idx=77,
                    timestamp_ms=400,
                    folder_id="folder-a",
                    caption="a person running",
                    objects={"person": 1},
                ),
                FilterResult(
                    frame_id="filter-2",
                    video_id="prefix.video-b",
                    frame_idx=88,
                    timestamp_ms=500,
                    folder_id="folder-a",
                    caption="a person waving",
                    objects={"person": 2},
                ),
            ],
        )


class _VbsLogger:
    """Capture participant-bound QueryResultLog payloads without DRES I/O."""

    def __init__(self, *, connected: bool = True, fail: bool = False) -> None:
        self.settings = DresSettings(base_url="https://dres.test")
        self.connected = connected
        self.fail = fail
        self.users: list[str] = []
        self.evaluations: list[str] = []
        self.payloads: list[Any] = []

    def session_status(self, user_id: str) -> dict[str, bool | str]:
        return {"user_id": user_id, "connected": self.connected}

    async def resolve_evaluation(self, user_id: str) -> str:
        self.users.append(user_id)
        return "eval-live"

    async def log_results(self, user_id: str, evaluation_id: str, payload: Any) -> None:
        if self.fail:
            raise RuntimeError("private-session token must not appear in logs")
        self.users.append(user_id)
        self.evaluations.append(evaluation_id)
        self.payloads.append(payload)

    def media_item_name(self, video_id: str) -> str:
        """Apply the same exact leading-prefix mapping as the DRES service."""

        prefix = "prefix."
        return video_id[len(prefix):] if video_id.startswith(prefix) else video_id


def _search_result(video_id: str, timestamp_ms: int) -> SearchResult:
    """Build a canonical representative frame with no DRES-specific fields."""

    return SearchResult(
        frame_id=f"frame-{timestamp_ms}",
        video_id=video_id,
        frame_idx=999,
        timestamp_ms=timestamp_ms,
        score=0.5,
        frame_ids=[f"frame-{timestamp_ms}"],
        timestamps_ms=[timestamp_ms],
        metadata=SearchResultMetadata(),
    )


def _jpeg() -> bytes:
    """Build one valid tiny query image in memory."""

    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(stream, format="JPEG")
    return stream.getvalue()


def _client(service: _SearchService, vbs: _VbsLogger | None):
    """Create an isolated ASGI client with frozen event wall-clock time."""

    app = FastAPI()
    container = {"service": service, "vbs_service": vbs}
    app.include_router(create_kis_router(container))
    app.include_router(create_search_router(container))
    transport = httpx.ASGITransport(app=app)
    return app, transport


def test_text_search_logs_full_ranked_results_and_trimmed_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the performing participant's session and 1-based result order."""

    monkeypatch.setattr("hcmai.api.routers.kis._now_ms", lambda: EPOCH_MS)
    logger = _VbsLogger()
    _, transport = _client(_SearchService(), logger)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/kis/search",
                headers={"X-VBS-User-ID": "member-who-searched"},
                json={
                    "inputs": [{"text": "  person running  "}],
                    "expected_revision": 0,
                    "use_dense": True,
                    "use_bm25": True,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "sent"
    assert logger.users == ["member-who-searched", "member-who-searched"]
    assert logger.evaluations == ["eval-live"]
    assert logger.payloads[0].model_dump(by_alias=True, exclude_none=True) == {
        "timestamp": EPOCH_MS,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 1, "answer": {"mediaItemName": "video-a", "start": 100, "end": 100}},
            {"rank": 2, "answer": {"mediaItemName": "video-b", "start": 200, "end": 200}},
        ],
        "events": [{
            "timestamp": EPOCH_MS,
            "category": "TEXT",
            "type": "SEARCH",
            "value": "person running",
        }],
    }


def test_image_search_logs_filename_and_digest_without_image_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Log only upload identity metadata and complete visual results."""

    monkeypatch.setattr("hcmai.api.routers.search._now_ms", lambda: EPOCH_MS)
    logger = _VbsLogger()
    _, transport = _client(_SearchService(), logger)
    payload = _jpeg()
    digest = hashlib.sha256(payload).hexdigest()

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/search/image",
                headers={"X-VBS-User-ID": "member-image"},
                files={"image": ("query.jpg", payload, "image/jpeg")},
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "sent"
    assert logger.payloads[0].model_dump(by_alias=True, exclude_none=True) == {
        "timestamp": EPOCH_MS,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 1, "answer": {"mediaItemName": "video-a", "start": 300, "end": 300}},
        ],
        "events": [{
            "timestamp": EPOCH_MS,
            "category": "IMAGE",
            "type": "SEARCH",
            "value": f"filename=query.jpg;sha256={digest}",
        }],
    }
    assert digest not in response.text
    assert payload.decode("latin-1") not in response.text


def test_filter_page_two_logs_global_rank_and_stable_predicate_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve global filter ranks and exact frame times on page two."""

    monkeypatch.setattr("hcmai.api.routers.search._now_ms", lambda: EPOCH_MS)
    logger = _VbsLogger()
    _, transport = _client(_SearchService(), logger)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/filter",
                headers={"X-VBS-User-ID": "member-filter"},
                json={
                    "metadata_filters": {"caption": "person", "objects": {"person": 2}},
                    "folder_id": "folder-a",
                    "page_id": 2,
                    "frames_per_pages": 20,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "sent"
    assert logger.payloads[0].model_dump(by_alias=True, exclude_none=True) == {
        "timestamp": EPOCH_MS,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 21, "answer": {"mediaItemName": "video-a", "start": 400, "end": 400}},
            {"rank": 22, "answer": {"mediaItemName": "video-b", "start": 500, "end": 500}},
        ],
        "events": [{
            "timestamp": EPOCH_MS,
            "category": "FILTER",
            "type": "SEARCH",
            "value": '{"folder_id":"folder-a","metadata_filters":{"caption":"person","objects":{"person":2}}}',
        }],
    }


def test_connected_user_logs_successful_search_even_when_legacy_toggle_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connected participant's successful search always emits its result log."""

    monkeypatch.setattr("hcmai.api.routers.kis._now_ms", lambda: EPOCH_MS)
    logger = _VbsLogger()
    logger.settings = SimpleNamespace(logging_enabled=False)
    _, transport = _client(_SearchService(), logger)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/kis/search",
                headers={"X-VBS-User-ID": "member-a"},
                json={
                    "inputs": [{"text": "person running"}],
                    "expected_revision": 0,
                    "use_dense": True,
                    "use_bm25": True,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "sent"
    assert logger.users == ["member-a", "member-a"]
    assert len(logger.payloads) == 1


def test_logging_failure_does_not_change_retrieval_success_or_leak_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep successful retrieval visible when result logging raises."""

    monkeypatch.setattr("hcmai.api.routers.kis._now_ms", lambda: EPOCH_MS)
    _, transport = _client(_SearchService(), _VbsLogger(fail=True))

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/kis/search",
                headers={"X-VBS-User-ID": "member-a"},
                json={
                    "inputs": [{"text": "person running"}],
                    "expected_revision": 0,
                    "use_dense": True,
                    "use_bm25": True,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "failed"
    assert response.json()["results"][0]["video_id"] == "video-a"
    assert "private-session" not in response.text


@pytest.mark.parametrize("user_id,connected", [(None, False), ("member-a", False)])
def test_unconfigured_or_unconnected_logging_is_skipped_without_blocking_search(
    user_id: str | None,
    connected: bool,
) -> None:
    """Keep local retrieval available when DRES logging cannot be attempted."""

    vbs = _VbsLogger(connected=connected)
    _, transport = _client(_SearchService(), vbs)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {} if user_id is None else {"X-VBS-User-ID": user_id}
            return await client.post(
                "/api/v1/kis/search",
                headers=headers,
                json={
                    "inputs": [{"text": "person"}],
                    "expected_revision": 0,
                    "use_dense": True,
                    "use_bm25": True,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "skipped"
    assert vbs.payloads == []


def test_absent_dres_service_skips_logging_and_returns_search_results() -> None:
    """Keep local searches healthy when server-side DRES config is absent."""

    _, transport = _client(_SearchService(), None)

    async def send() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/v1/kis/search",
                headers={"X-VBS-User-ID": "member-a"},
                json={
                    "inputs": [{"text": "person"}],
                    "expected_revision": 0,
                    "use_dense": True,
                    "use_bm25": True,
                },
            )

    response = asyncio.run(send())

    assert response.status_code == 200
    assert response.headers["x-dres-log-status"] == "skipped"
    assert response.json()["results"][0]["video_id"] == "video-a"
