"""HTTP contracts for the integrated local-video FastAPI router."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from hcmai.app import create_app
from hcmai.common import environment
from hcmai.orchestration.pipeline import SearchService
from hcmai.socketapp.catalog import VideoCatalog
from hcmai.api.routers.videos import (
    ByteRange,
    RangeNotSatisfiable,
    parse_byte_range,
)


def request(app: FastAPI, method: str, path: str, **kwargs: object) -> httpx.Response:
    """Issue one request through the ASGI boundary without a live server."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(send())
    finally:
        loop.close()


@pytest.fixture()
def video_app(tmp_path: Path) -> FastAPI:
    """Create the main HCMAI app with one injected local source video."""

    root = tmp_path / "videos"
    root.mkdir()
    (root / "sample.mp4").write_bytes(b"0123456789abcdef")
    return create_app(video_catalog=VideoCatalog(root))


def test_parse_byte_ranges() -> None:
    """Open-ended and suffix ranges are clamped to the representation size."""

    assert parse_byte_range("bytes=2-5", 10) == ByteRange(2, 5)
    assert parse_byte_range("bytes=2-", 10) == ByteRange(2, 9)
    assert parse_byte_range("bytes=-3", 10) == ByteRange(7, 9)
    assert parse_byte_range("items=2-5", 10) is None
    with pytest.raises(RangeNotSatisfiable):
        parse_byte_range("bytes=10-", 10)
    with pytest.raises(RangeNotSatisfiable):
        parse_byte_range("bytes=0-1,3-4", 10)


def test_get_and_range_stream_have_media_headers(video_app: FastAPI) -> None:
    """The main app serves complete and partial media representations."""

    response = request(
        video_app,
        "GET",
        "/api/v1/videos/sample/stream",
        headers={"Origin": "http://localhost:3000"},
    )
    assert response.status_code == 200
    assert response.content == b"0123456789abcdef"
    assert response.headers["content-length"] == "16"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    response = request(
        video_app,
        "GET",
        "/api/v1/videos/sample/stream",
        headers={"Range": "bytes=4-9"},
    )
    assert response.status_code == 206
    assert response.content == b"456789"
    assert response.headers["content-range"] == "bytes 4-9/16"
    assert response.headers["content-length"] == "6"


def test_health_list_and_metadata_do_not_expose_paths(video_app: FastAPI) -> None:
    """Operational endpoints expose canonical IDs but not origin paths."""

    response = request(video_app, "GET", "/api/v1/videos/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hcmai-video-origin",
        "ready": True,
        "video_count": 1,
    }

    response = request(video_app, "GET", "/api/v1/videos")
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert "/tmp" not in response.text
    assert "sample.mp4" not in response.text

    response = request(video_app, "GET", "/api/v1/videos/sample")
    assert response.status_code == 200
    assert response.json()["stream_url"] == "/api/v1/videos/sample/stream"
    assert response.json()["player_url"] == (
        "/api/v1/videos/sample/play?timestamp_ms=0"
    )
    assert "sample.mp4" not in response.text


def test_unconfigured_catalog_has_explicit_health_and_503() -> None:
    """Register video routes even when local source videos are not configured."""

    app = create_app()

    health = request(app, "GET", "/api/v1/videos/health")
    assert health.status_code == 200
    assert health.json()["ready"] is False
    response = request(app, "GET", "/api/v1/videos")
    assert response.status_code == 503
    assert response.json()["detail"] == "Local video catalog is not configured"


def test_app_lifespan_loads_catalog_from_repository_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main app builds the video catalog from the repository ``.env``."""

    root = tmp_path / "videos"
    root.mkdir()
    (root / "L21_V001.mp4").write_bytes(b"video")
    (tmp_path / ".env").write_text(
        f"SOCKETAPP_VIDEO_ROOT={root}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SOCKETAPP_VIDEO_ROOT", raising=False)
    monkeypatch.setattr(environment, "REPOSITORY_ROOT", tmp_path)
    app = create_app(search_service=SearchService(corpus=None, retrieval=None))

    async def verify() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/v1/videos")
        assert response.status_code == 200
        assert response.json()["videos"][0]["video_id"] == "L21_V001"

    asyncio.run(verify())


def test_player_accepts_milliseconds_and_head(video_app: FastAPI) -> None:
    """The integrated player preserves millisecond seeking and HEAD semantics."""

    response = request(
        video_app,
        "GET",
        "/api/v1/videos/sample/play?timestamp_ms=1250",
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "const requestedMs = 1250;" in response.text
    assert "requestVideoFrameCallback" in response.text
    assert "/api/v1/videos/sample/stream" in response.text

    response = request(
        video_app,
        "HEAD",
        "/api/v1/videos/sample/play?timestamp_ms=1250",
    )
    assert response.status_code == 200
    assert response.content == b""
    assert int(response.headers["content-length"]) > 0


@pytest.mark.parametrize("query", ["timestamp_ms=", "timestamp_ms=-1", "timestamp_ms=1.5"])
def test_player_rejects_invalid_timestamps(
    video_app: FastAPI, query: str
) -> None:
    """Malformed timestamps never reach the player renderer."""

    response = request(
        video_app,
        "GET",
        f"/api/v1/videos/sample/play?{query}",
    )
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "timestamp_ms must be a non-negative integer"
    )


def test_head_cache_and_invalid_range_contracts(video_app: FastAPI) -> None:
    """HEAD, revalidation, and unsatisfied ranges have deterministic bodies."""

    response = request(video_app, "HEAD", "/api/v1/videos/sample/stream")
    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-length"] == "16"

    response = request(video_app, "GET", "/api/v1/videos/sample/stream")
    etag = response.headers["etag"]
    response = request(
        video_app,
        "GET",
        "/api/v1/videos/sample/stream",
        headers={"If-None-Match": etag},
    )
    assert response.status_code == 304
    assert response.content == b""

    response = request(
        video_app,
        "GET",
        "/api/v1/videos/sample/stream",
        headers={"Range": "bytes=999-1000"},
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */16"
    assert "not satisfiable" in response.text


def test_unknown_video_and_paths_never_become_filesystem_lookups(
    video_app: FastAPI,
) -> None:
    """Unknown canonical IDs remain inside the declared route contract."""

    response = request(
        video_app,
        "GET",
        "/api/v1/videos/does-not-exist/stream",
    )
    assert response.status_code == 404
    response = request(
        video_app,
        "GET",
        "/api/v1/videos/../sample.mp4/stream",
    )
    assert response.status_code in {400, 404}
