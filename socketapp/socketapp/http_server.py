"""HTTP/1.1 byte-range server for cataloged local videos.

The transport owns browser-facing HTTP semantics only. It serves one range at
a time, sends deterministic ``Content-Length``/``Content-Range`` headers, and
uses ``socket.sendfile`` for regular files when the platform supports it. It
does not implement HLS transcoding, arbitrary filesystem browsing, uploads, or
Cloudflare authentication.
"""

from __future__ import annotations

import email.utils
import json
import logging
import os
import re
import socket
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .catalog import VideoCatalog, VideoEntry
from .config import Settings

LOGGER = logging.getLogger("socketapp.http")
RANGE_RE = re.compile(r"^(\d*)-(\d*)$")
MAX_RANGE_HEADER_BYTES = 256
MAX_TIMESTAMP_MS = 9_007_199_254_740_991
CHUNK_SIZE = 1024 * 1024


class RangeNotSatisfiable(ValueError):
    """Raised when a single byte range cannot be served."""


@dataclass(frozen=True, slots=True)
class ByteRange:
    """Inclusive byte range selected for one partial response."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of bytes in this inclusive range."""

        return self.end - self.start + 1


def parse_byte_range(value: str, size: int) -> ByteRange | None:
    """Parse a single RFC 9110 ``bytes`` range.

    Unknown range units are ignored so the caller can return a normal ``200``
    response. Multiple byte ranges are intentionally rejected; browser media
    elements request single ranges, and a multipart implementation would add
    buffering and parsing complexity without helping this use case.
    """

    if not value or len(value.encode("ascii", errors="ignore")) > MAX_RANGE_HEADER_BYTES:
        raise RangeNotSatisfiable("range header is empty or too long")
    unit, separator, range_spec = value.partition("=")
    if not separator or unit.strip().lower() != "bytes":
        return None
    if "," in range_spec:
        raise RangeNotSatisfiable("multiple byte ranges are not supported")
    match = RANGE_RE.fullmatch(range_spec.strip())
    if match is None or size <= 0:
        raise RangeNotSatisfiable("malformed or empty byte range")

    raw_start, raw_end = match.groups()
    if not raw_start and not raw_end:
        raise RangeNotSatisfiable("empty byte range")
    if not raw_start:
        suffix_length = int(raw_end)
        if suffix_length <= 0:
            raise RangeNotSatisfiable("suffix range must be positive")
        start = max(size - suffix_length, 0)
        return ByteRange(start, size - 1)

    start = int(raw_start)
    if start >= size:
        raise RangeNotSatisfiable("range starts after representation")
    end = size - 1 if not raw_end else min(int(raw_end), size - 1)
    if end < start:
        raise RangeNotSatisfiable("range ends before it starts")
    return ByteRange(start, end)


def _etag(stat_result: os.stat_result) -> str:
    """Create a strong-enough local-file validator for cache revalidation."""

    return (
        f'"{stat_result.st_mtime_ns:x}-{stat_result.st_size:x}-'
        f'{getattr(stat_result, "st_ino", 0):x}"'
    )


def _http_date(timestamp: float) -> str:
    """Format a filesystem timestamp as an RFC-compatible HTTP date."""

    return email.utils.formatdate(timestamp, usegmt=True)


def _parse_http_date(value: str) -> datetime | None:
    """Parse an HTTP date into an aware UTC datetime, if valid."""

    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _if_range_matches(value: str | None, etag: str, modified: float) -> bool:
    """Return whether a client validator permits a requested byte range."""

    if value is None:
        return True
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.startswith('W/'):
        return False
    if candidate.startswith('"'):
        return candidate == etag
    parsed = _parse_http_date(candidate)
    if parsed is None:
        return False
    return parsed.timestamp() >= int(modified)


def _matches_not_modified(value: str | None, etag: str) -> bool:
    """Return whether an ``If-None-Match`` value matches this file."""

    if value is None:
        return False
    return any(item.strip() in {etag, "*"} for item in value.split(","))


def _matches_since(value: str | None, modified: float) -> bool:
    """Return whether ``If-Modified-Since`` says the file is unchanged."""

    if value is None:
        return False
    parsed = _parse_http_date(value)
    return parsed is not None and int(modified) <= int(parsed.timestamp())


class VideoRequestHandler(BaseHTTPRequestHandler):
    """Serve health, catalog, playback, and local video stream endpoints."""

    protocol_version = "HTTP/1.1"
    server_version = "SocketApp/0.1"

    def __init__(
        self,
        request: Any,
        client_address: Any,
        server: VideoHTTPServer,
    ) -> None:
        """Attach the catalog and immutable settings supplied by the server."""

        self.catalog = server.catalog
        self.settings = server.settings
        super().__init__(request, client_address, server)

    def setup(self) -> None:
        """Apply bounded socket read/write timeouts and low-latency TCP mode."""

        super().setup()
        self.connection.settimeout(self.settings.request_timeout_seconds)
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # Non-TCP test transports and unusual platforms may not expose this.
            pass

    def log_message(self, format: str, *args: Any) -> None:
        """Log method and path without query strings or header/token values."""

        path = urlsplit(self.path).path
        LOGGER.info("%s %s %s", self.address_string(), self.command, path)

    def do_OPTIONS(self) -> None:
        """Answer browser CORS preflight requests without touching the catalog."""

        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        """Route one GET request to a JSON or byte-stream response."""

        self._dispatch(head_only=False)

    def do_HEAD(self) -> None:
        """Route one HEAD request while suppressing response bodies."""

        self._dispatch(head_only=True)

    def do_POST(self) -> None:
        """Reject mutation requests because this origin is read-only."""

        self._json_error(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")

    def _dispatch(self, *, head_only: bool) -> None:
        """Resolve a path and call its narrow handler."""

        try:
            path = urlsplit(self.path).path
        except ValueError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid request target")
            return

        if path in {"/", "/health", "/ready"}:
            self._health(path)
            return
        if path == "/api/v1/videos":
            self._list_videos()
            return

        prefix = "/api/v1/videos/"
        if not path.startswith(prefix):
            self._json_error(HTTPStatus.NOT_FOUND, "route not found")
            return

        suffix = path[len(prefix):]
        if suffix.endswith("/stream"):
            encoded_id = suffix[: -len("/stream")].rstrip("/")
            video_id = self._decode_id(encoded_id)
            if video_id is None:
                return
            self._stream_video(video_id, head_only=head_only)
            return

        if suffix.endswith("/play"):
            encoded_id = suffix[: -len("/play")].rstrip("/")
            video_id = self._decode_id(encoded_id)
            if video_id is None:
                return
            try:
                timestamp_ms = self._timestamp_ms(urlsplit(self.path).query)
            except ValueError as error:
                self._json_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._play_video(video_id, timestamp_ms, head_only=head_only)
            return

        video_id = self._decode_id(suffix.rstrip("/"))
        if video_id is None:
            return
        self._video_metadata(video_id)

    def _decode_id(self, encoded_id: str) -> str | None:
        """Decode exactly one URL segment and reject encoded path separators."""

        if not encoded_id or "/" in encoded_id:
            self._json_error(HTTPStatus.NOT_FOUND, "video not found")
            return None
        try:
            video_id = unquote(encoded_id, errors="strict")
        except UnicodeDecodeError:
            self._json_error(HTTPStatus.BAD_REQUEST, "invalid video ID")
            return None
        if not video_id or "/" in video_id or "\\" in video_id or "\x00" in video_id:
            self._json_error(HTTPStatus.NOT_FOUND, "video not found")
            return None
        return video_id

    @staticmethod
    def _timestamp_ms(query: str) -> int:
        """Parse one non-negative, JavaScript-safe ``timestamp_ms`` value."""

        values = parse_qs(query, keep_blank_values=True).get("timestamp_ms")
        if values is None:
            return 0
        if len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]):
            raise ValueError("timestamp_ms must be a non-negative integer")
        timestamp_ms = int(values[0])
        if timestamp_ms > MAX_TIMESTAMP_MS:
            raise ValueError("timestamp_ms is too large")
        return timestamp_ms

    def _health(self, path: str) -> None:
        """Return process/readiness state without exposing local filesystem paths."""

        ready = len(self.catalog) > 0
        payload = {
            "status": "ok" if ready else "degraded",
            "service": "hcmai-socketapp",
            "ready": ready,
            "video_count": len(self.catalog),
        }
        status = HTTPStatus.OK if ready or path != "/ready" else HTTPStatus.SERVICE_UNAVAILABLE
        self._json_response(status, payload, cache_control="no-store")

    def _list_videos(self) -> None:
        """Return lightweight catalog metadata without local paths."""

        videos = []
        for entry in self.catalog.entries():
            try:
                stat_result = entry.snapshot()
            except OSError:
                # A file removed after startup is not advertised as playable.
                continue
            videos.append(self._metadata_payload(entry, stat_result))
        self._json_response(
            HTTPStatus.OK,
            {"videos": videos, "count": len(videos)},
            cache_control="no-store",
        )

    def _video_metadata(self, video_id: str) -> None:
        """Return metadata and the stream URL for one exact video ID."""

        entry = self.catalog.get(video_id)
        if entry is None:
            self._json_error(HTTPStatus.NOT_FOUND, "video not found")
            return
        try:
            stat_result = entry.snapshot()
        except OSError:
            self._json_error(HTTPStatus.NOT_FOUND, "video is unavailable")
            return
        self._json_response(
            HTTPStatus.OK,
            self._metadata_payload(entry, stat_result),
            cache_control="no-store",
        )

    def _metadata_payload(
        self, entry: VideoEntry, stat_result: os.stat_result
    ) -> dict[str, object]:
        """Build public metadata while hiding the origin's absolute path."""

        encoded_id = quote(entry.video_id, safe="")
        return {
            "video_id": entry.video_id,
            "size_bytes": stat_result.st_size,
            "media_type": entry.media_type,
            "fps": entry.fps,
            "stream_url": f"/api/v1/videos/{encoded_id}/stream",
            "player_url": f"/api/v1/videos/{encoded_id}/play?timestamp_ms=0",
            "etag": _etag(stat_result),
            "last_modified": _http_date(stat_result.st_mtime),
        }

    def _play_video(
        self, video_id: str, timestamp_ms: int, *, head_only: bool
    ) -> None:
        """Render a standalone player that seeks and displays milliseconds."""

        entry = self.catalog.get(video_id)
        if entry is None:
            self._json_error(HTTPStatus.NOT_FOUND, "video not found")
            return
        try:
            entry.snapshot()
        except OSError:
            self._json_error(HTTPStatus.NOT_FOUND, "video is unavailable")
            return

        from .player import render_player

        encoded_id = quote(entry.video_id, safe="")
        body = render_player(
            video_id=entry.video_id,
            media_type=entry.media_type,
            stream_path=f"/api/v1/videos/{encoded_id}/stream",
            timestamp_ms=timestamp_ms,
            source_fps=entry.fps,
        )
        self._html_response(body, head_only=head_only)

    def _stream_video(self, video_id: str, *, head_only: bool) -> None:
        """Serve one complete or partial local file with browser media headers."""

        entry = self.catalog.get(video_id)
        if entry is None:
            self._json_error(HTTPStatus.NOT_FOUND, "video not found")
            return

        try:
            video_file = self._open_video(entry)
        except (FileNotFoundError, PermissionError, OSError):
            self._json_error(HTTPStatus.NOT_FOUND, "video is unavailable")
            return

        with video_file:
            try:
                stat_result = os.fstat(video_file.fileno())
                if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size < 0:
                    raise OSError("video file is not regular")
                size = stat_result.st_size
                etag = _etag(stat_result)
                modified = stat_result.st_mtime

                if _matches_not_modified(self.headers.get("If-None-Match"), etag):
                    self._send_not_modified(etag, modified)
                    return
                if (
                    self.headers.get("If-None-Match") is None
                    and _matches_since(self.headers.get("If-Modified-Since"), modified)
                ):
                    self._send_not_modified(etag, modified)
                    return

                requested_range = self.headers.get("Range")
                selected_range = None
                if requested_range and _if_range_matches(
                    self.headers.get("If-Range"), etag, modified
                ):
                    try:
                        selected_range = parse_byte_range(requested_range, size)
                    except RangeNotSatisfiable:
                        self._range_error(size, etag, modified)
                        return

                if selected_range is None:
                    status = HTTPStatus.OK
                    start = 0
                    length = size
                else:
                    status = HTTPStatus.PARTIAL_CONTENT
                    start = selected_range.start
                    length = selected_range.length

                self.send_response(status)
                self._send_common_headers()
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", entry.media_type)
                self.send_header("Content-Length", str(length))
                self.send_header("ETag", etag)
                self.send_header("Last-Modified", _http_date(modified))
                self.send_header(
                    "Content-Disposition",
                    f"inline; filename*=UTF-8''{quote(entry.path.name, safe='')}",
                )
                if selected_range is not None:
                    self.send_header(
                        "Content-Range",
                        f"bytes {selected_range.start}-{selected_range.end}/{size}",
                    )
                self.end_headers()
                if not head_only and length and self._send_file(video_file, start, length) != length:
                    # Do not leave a keep-alive connection waiting for a
                    # Content-Length that could not be delivered after a
                    # concurrent truncate or storage error.
                    self.close_connection = True
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                # The browser is allowed to cancel a range while seeking. Once
                # headers are out, there is no useful second response to send.
                self.close_connection = True
                LOGGER.debug("video transfer ended early video_id=%s", video_id)

    @staticmethod
    def _open_video(entry: VideoEntry):
        """Open a cataloged file read-only, rejecting a replaced final symlink."""

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(entry.path, flags)
        return os.fdopen(descriptor, "rb", buffering=CHUNK_SIZE)

    def _send_file(self, video_file: Any, start: int, length: int) -> int:
        """Transfer a bounded file slice and return bytes actually delivered."""

        if hasattr(self.connection, "sendfile"):
            offset = start
            remaining = length
            while remaining > 0:
                sent = self.connection.sendfile(video_file, offset, remaining)
                if not sent:
                    break
                offset += sent
                remaining -= sent
            return length - remaining

        video_file.seek(start)
        remaining = length
        while remaining > 0:
            chunk = video_file.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)
        return length - remaining

    def _range_error(
        self, size: int, etag: str, modified: float
    ) -> None:
        """Return a standards-compatible unsatisfied-range response."""

        self.send_response(HTTPStatus.RANGE_NOT_SATISFIABLE)
        self._send_common_headers()
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", _http_date(modified))
        body = json.dumps(
            {"detail": "requested byte range is not satisfiable"},
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_not_modified(self, etag: str, modified: float) -> None:
        """Send a bodyless cache revalidation response."""

        self.send_response(HTTPStatus.NOT_MODIFIED)
        self._send_common_headers()
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", _http_date(modified))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json_response(
        self,
        status: HTTPStatus,
        payload: object,
        *,
        cache_control: str = "no-store",
    ) -> None:
        """Serialize one JSON response with explicit length and CORS headers."""

        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _html_response(self, body: bytes, *, head_only: bool) -> None:
        """Return a self-contained player page without executing local code."""

        self.send_response(HTTPStatus.OK)
        self._send_common_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; media-src 'self'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _json_error(self, status: HTTPStatus, detail: str) -> None:
        """Return a minimal JSON error without internal paths or exceptions."""

        self._json_response(status, {"detail": detail})

    def _send_common_headers(self) -> None:
        """Add transport and CORS headers shared by every response."""

        origin = self.headers.get("Origin")
        origins = self.settings.cors_origins
        if "*" in origins:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif origin and origin in origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Range, Content-Type, If-Range"
        )
        self.send_header(
            "Access-Control-Expose-Headers",
            "Accept-Ranges, Content-Length, Content-Range, Content-Type, ETag, Last-Modified",
        )
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("X-Content-Type-Options", "nosniff")


class _BoundedThreadingMixIn(ThreadingMixIn):
    """Bound request threads so stalled clients cannot exhaust the host."""

    daemon_threads = True
    block_on_close = True
    request_queue_size = 128

    def __init__(self, *args: Any, max_workers: int, **kwargs: Any) -> None:
        """Create a fixed admission budget for active HTTP requests."""

        self._request_slots = threading.BoundedSemaphore(max_workers)
        self._max_workers = max_workers
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        """Admit requests while capacity exists and close excess sockets."""

        if not self._request_slots.acquire(blocking=False):
            LOGGER.warning("request capacity exhausted; closing %s", client_address[0])
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Release the request slot after the handler thread exits."""

        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class VideoHTTPServer(_BoundedThreadingMixIn, HTTPServer):
    """Thread-bounded HTTP server bound to one validated ``VideoCatalog``."""

    allow_reuse_address = True

    def __init__(self, settings: Settings, catalog: VideoCatalog) -> None:
        """Bind the configured host and create request handlers."""

        self.settings = settings
        self.catalog = catalog
        super().__init__(
            (settings.host, settings.port),
            VideoRequestHandler,
            max_workers=settings.max_workers,
        )

    def server_bind(self) -> None:
        """Set keepalive before binding the listening socket."""

        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        super().server_bind()
