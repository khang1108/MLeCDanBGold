"""Focused HTTP contract and byte-range tests for SocketApp."""

from __future__ import annotations

import http.client
import threading
from pathlib import Path

import pytest

from socketapp.catalog import VideoCatalog
from socketapp.config import Settings
from socketapp.http_server import (
    ByteRange,
    RangeNotSatisfiable,
    VideoHTTPServer,
    parse_byte_range,
)


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


@pytest.fixture()
def running_server(tmp_path: Path):
    """Start a real loopback server and clean it up after each test."""

    root = tmp_path / "videos"
    root.mkdir()
    (root / "sample.mp4").write_bytes(b"0123456789abcdef")
    catalog = VideoCatalog(root)
    settings = Settings(
        host="127.0.0.1",
        port=0,
        video_root=root,
        max_workers=4,
        cors_origins=("http://frontend.test",),
    )
    server = VideoHTTPServer(settings, catalog)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server: VideoHTTPServer, method: str, path: str, **headers):
    """Issue one HTTP request to the fixture server."""

    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=3)
    connection.request(method, path, headers=headers)
    response = connection.getresponse()
    body = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response, response_headers, body


def test_get_and_range_stream_have_media_headers(running_server: VideoHTTPServer) -> None:
    """Native media clients receive complete and partial representations."""

    response, headers, body = request(
        running_server,
        "GET",
        "/api/v1/videos/sample/stream",
        Origin="http://frontend.test",
    )
    assert response.status == 200
    assert body == b"0123456789abcdef"
    assert headers["content-length"] == "16"
    assert headers["content-type"] == "video/mp4"
    assert headers["accept-ranges"] == "bytes"
    assert headers["access-control-allow-origin"] == "http://frontend.test"

    response, headers, body = request(
        running_server,
        "GET",
        "/api/v1/videos/sample/stream",
        Range="bytes=4-9",
    )
    assert response.status == 206
    assert body == b"456789"
    assert headers["content-range"] == "bytes 4-9/16"
    assert headers["content-length"] == "6"


def test_health_list_and_metadata_do_not_expose_origin_paths(
    running_server: VideoHTTPServer,
) -> None:
    """Operational endpoints expose IDs and sizes, never local path names."""

    response, _, body = request(running_server, "GET", "/ready")
    assert response.status == 200
    assert body == b'{"status":"ok","service":"hcmai-socketapp","ready":true,"video_count":1}'

    response, _, body = request(running_server, "GET", "/api/v1/videos")
    assert response.status == 200
    assert b'"video_id":"sample"' in body
    assert b"/tmp" not in body
    assert b"sample.mp4" not in body

    response, _, body = request(running_server, "GET", "/api/v1/videos/sample")
    assert response.status == 200
    assert b'"stream_url":"/api/v1/videos/sample/stream"' in body
    assert b'"player_url":"/api/v1/videos/sample/play?timestamp_ms=0"' in body
    assert b"sample.mp4" not in body


def test_player_accepts_milliseconds_and_tracks_rendered_media_time(
    running_server: VideoHTTPServer,
) -> None:
    """The standalone player seeks by integer milliseconds and exposes a clock."""

    response, headers, body = request(
        running_server,
        "GET",
        "/api/v1/videos/sample/play?timestamp_ms=1250",
    )
    assert response.status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert b"const requestedMs = 1250;" in body
    assert b"requestVideoFrameCallback" in body
    assert b"Current timestamp:" in body
    assert b"/api/v1/videos/sample/stream" in body

    response, headers, body = request(
        running_server,
        "HEAD",
        "/api/v1/videos/sample/play?timestamp_ms=1250",
    )
    assert response.status == 200
    assert body == b""
    assert int(headers["content-length"]) > 0


@pytest.mark.parametrize("query", ["timestamp_ms=", "timestamp_ms=-1", "timestamp_ms=1.5"])
def test_player_rejects_non_integer_millisecond_timestamps(
    running_server: VideoHTTPServer, query: str
) -> None:
    """Malformed timestamp values never reach the player renderer."""

    response, _, body = request(
        running_server,
        "GET",
        f"/api/v1/videos/sample/play?{query}",
    )
    assert response.status == 400
    assert b"timestamp_ms must be a non-negative integer" in body


def test_head_cache_and_invalid_range_contracts(running_server: VideoHTTPServer) -> None:
    """HEAD, revalidation, and failed ranges do not send misleading bodies."""

    response, headers, body = request(
        running_server, "HEAD", "/api/v1/videos/sample/stream"
    )
    assert response.status == 200
    assert body == b""
    assert headers["content-length"] == "16"

    response, headers, body = request(
        running_server, "GET", "/api/v1/videos/sample/stream"
    )
    assert response.status == 200
    etag = headers["etag"]
    response, headers, body = request(
        running_server,
        "GET",
        "/api/v1/videos/sample/stream",
        **{"If-None-Match": etag},
    )
    assert response.status == 304
    assert body == b""

    response, headers, body = request(
        running_server,
        "GET",
        "/api/v1/videos/sample/stream",
        Range="bytes=999-1000",
    )
    assert response.status == 416
    assert headers["content-range"] == "bytes */16"
    assert b"not satisfiable" in body


def test_unknown_video_and_path_are_not_filesystem_lookups(
    running_server: VideoHTTPServer,
) -> None:
    """Unknown IDs and extra path segments stay inside the route contract."""

    response, _, _ = request(
        running_server, "GET", "/api/v1/videos/does-not-exist/stream"
    )
    assert response.status == 404
    response, _, _ = request(
        running_server, "GET", "/api/v1/videos/../sample.mp4/stream"
    )
    assert response.status in {404, 400}
