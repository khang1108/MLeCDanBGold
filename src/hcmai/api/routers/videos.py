"""FastAPI routes for catalog metadata, playback, and byte-range streaming.

The router serves only files resolved by :class:`VideoCatalog`. It preserves
browser range/cache semantics without exposing arbitrary filesystem lookup,
and leaves application startup and Cloudflare routing to ``hcmai.app``.
"""

from __future__ import annotations

import email.utils
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, BinaryIO
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse

from hcmai.socketapp.catalog import VideoCatalog, VideoEntry
from hcmai.socketapp.player import render_player


RANGE_RE = re.compile(r"^(\d*)-(\d*)$")
MAX_RANGE_HEADER_BYTES = 256
MAX_TIMESTAMP_MS = 9_007_199_254_740_991
CHUNK_SIZE = 1024 * 1024


class RangeNotSatisfiable(ValueError):
    """Raised when a requested single byte range cannot be served."""


@dataclass(frozen=True, slots=True)
class ByteRange:
    """Inclusive byte range selected for one partial response."""

    start: int
    end: int

    @property
    def length(self) -> int:
        """Return the number of bytes in this inclusive range."""

        return self.end - self.start + 1


class _VideoStreamResponse(Response):
    """Send an already-open bounded file slice through the ASGI transport."""

    def __init__(
        self,
        handle: BinaryIO,
        *,
        start: int,
        length: int,
        status_code: int,
        headers: dict[str, str],
        media_type: str,
        head_only: bool,
    ) -> None:
        """Retain the validated descriptor until the response is completed."""

        self.handle = handle
        self.start = start
        self.length = length
        self.head_only = head_only
        super().__init__(
            content=b"",
            status_code=status_code,
            headers=headers,
            media_type=media_type,
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Write response frames while never reading beyond the selected range."""

        del scope, receive
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            if self.head_only:
                await send({"type": "http.response.body", "body": b""})
                return

            self.handle.seek(self.start)
            remaining = self.length
            while remaining > 0:
                chunk = self.handle.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk,
                        "more_body": remaining > 0,
                    }
                )
            if self.length == 0 or remaining > 0:
                await send({"type": "http.response.body", "body": b""})
        finally:
            self.handle.close()


def parse_byte_range(value: str, size: int) -> ByteRange | None:
    """Parse one RFC 9110 byte range and reject multipart requests."""

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
        return ByteRange(max(size - suffix_length, 0), size - 1)

    start = int(raw_start)
    if start >= size:
        raise RangeNotSatisfiable("range starts after representation")
    end = size - 1 if not raw_end else min(int(raw_end), size - 1)
    if end < start:
        raise RangeNotSatisfiable("range ends before it starts")
    return ByteRange(start, end)


def _catalog(container: dict[str, Any]) -> VideoCatalog:
    """Return the initialized catalog or an HTTP 503 response."""

    catalog = container.get("video_catalog")
    if catalog is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local video catalog is not configured",
        )
    return catalog


def _entry(container: dict[str, Any], video_id: str) -> VideoEntry:
    """Resolve one exact canonical video ID without filesystem fallback."""

    if any(character in video_id for character in ("/", "\\", "\x00")):
        raise HTTPException(status_code=404, detail="Video not found")
    entry = _catalog(container).get(video_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return entry


def _snapshot(entry: VideoEntry) -> os.stat_result:
    """Read current file metadata while translating disappearance to HTTP 404."""

    try:
        return entry.snapshot()
    except OSError:
        raise HTTPException(status_code=404, detail="Video is unavailable") from None


def _etag(stat_result: os.stat_result) -> str:
    """Build a local-file validator from modification, size, and inode."""

    return (
        f'"{stat_result.st_mtime_ns:x}-{stat_result.st_size:x}-'
        f'{getattr(stat_result, "st_ino", 0):x}"'
    )


def _http_date(timestamp: float) -> str:
    """Format a filesystem timestamp as an HTTP date."""

    return email.utils.formatdate(timestamp, usegmt=True)


def _parse_http_date(value: str) -> datetime | None:
    """Parse an HTTP date into UTC, returning ``None`` for invalid input."""

    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _matches_not_modified(value: str | None, etag: str) -> bool:
    """Return whether ``If-None-Match`` matches the current file."""

    return value is not None and any(
        item.strip() in {etag, "*"} for item in value.split(",")
    )


def _matches_since(value: str | None, modified: float) -> bool:
    """Return whether ``If-Modified-Since`` says the file is unchanged."""

    if value is None:
        return False
    parsed = _parse_http_date(value)
    return parsed is not None and int(modified) <= int(parsed.timestamp())


def _if_range_matches(value: str | None, etag: str, modified: float) -> bool:
    """Return whether an If-Range validator permits a partial response."""

    if value is None:
        return True
    candidate = value.strip()
    if not candidate or candidate.startswith("W/"):
        return False
    if candidate.startswith('"'):
        return candidate == etag
    parsed = _parse_http_date(candidate)
    return parsed is not None and parsed.timestamp() >= int(modified)


def _metadata(entry: VideoEntry, stat_result: os.stat_result) -> dict[str, object]:
    """Build public metadata while keeping the local path private."""

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


def _open_video(entry: VideoEntry) -> tuple[BinaryIO, os.stat_result]:
    """Open a cataloged regular file read-only without following a final symlink."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(entry.path, flags)
        handle = os.fdopen(descriptor, "rb", buffering=CHUNK_SIZE)
        stat_result = os.fstat(handle.fileno())
        if not stat.S_ISREG(stat_result.st_mode):
            handle.close()
            raise OSError("video path is not a regular file")
        return handle, stat_result
    except OSError:
        raise HTTPException(status_code=404, detail="Video is unavailable") from None


def _timestamp_ms(request: Request) -> int:
    """Parse one non-negative JavaScript-safe player timestamp."""

    values = request.query_params.getlist("timestamp_ms")
    if not values:
        return 0
    if len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]):
        raise HTTPException(
            status_code=400,
            detail="timestamp_ms must be a non-negative integer",
        )
    value = int(values[0])
    if value > MAX_TIMESTAMP_MS:
        raise HTTPException(status_code=400, detail="timestamp_ms is too large")
    return value


def create_video_router(service_container: dict[str, Any]) -> APIRouter:
    """Create integrated local-video catalog, player, and stream routes."""

    router = APIRouter(prefix="/api/v1/videos", tags=["videos"])

    @router.get("/health")
    async def video_health() -> dict[str, object]:
        """Report video-catalog readiness without exposing local paths."""

        catalog = service_container.get("video_catalog")
        count = 0 if catalog is None else len(catalog)
        return {
            "status": "ok" if count else "degraded",
            "service": "hcmai-video-origin",
            "ready": count > 0,
            "video_count": count,
        }

    @router.get("")
    async def list_videos() -> dict[str, object]:
        """List playable canonical IDs and public metadata."""

        videos = []
        for entry in _catalog(service_container).entries():
            try:
                videos.append(_metadata(entry, entry.snapshot()))
            except OSError:
                continue
        return {"videos": videos, "count": len(videos)}

    @router.get("/{video_id}")
    async def video_metadata(video_id: str) -> dict[str, object]:
        """Return metadata for one exact canonical video ID."""

        entry = _entry(service_container, video_id)
        return _metadata(entry, _snapshot(entry))

    @router.api_route("/{video_id}/play", methods=["GET", "HEAD"])
    async def play_video(video_id: str, request: Request) -> Response:
        """Render a self-contained browser player at a millisecond timestamp."""

        entry = _entry(service_container, video_id)
        _snapshot(entry)
        timestamp_ms = _timestamp_ms(request)
        encoded_id = quote(entry.video_id, safe="")
        body = render_player(
            video_id=entry.video_id,
            media_type=entry.media_type,
            stream_path=f"/api/v1/videos/{encoded_id}/stream",
            timestamp_ms=timestamp_ms,
            source_fps=entry.fps,
        )
        headers = {
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; media-src 'self'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'"
            ),
            "Referrer-Policy": "no-referrer",
        }
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(body))
            return Response(headers=headers, media_type="text/html")
        return HTMLResponse(content=body, headers=headers)

    @router.api_route("/{video_id}/stream", methods=["GET", "HEAD"])
    async def stream_video(video_id: str, request: Request) -> Response:
        """Serve a complete or single-range local video representation."""

        entry = _entry(service_container, video_id)
        handle, stat_result = _open_video(entry)
        size = stat_result.st_size
        etag = _etag(stat_result)
        modified = stat_result.st_mtime
        common_headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": service_container.get(
                "video_cache_control", "public, max-age=3600"
            ),
            "Content-Disposition": (
                f"inline; filename*=UTF-8''{quote(entry.path.name, safe='')}"
            ),
            "ETag": etag,
            "Last-Modified": _http_date(modified),
        }

        if _matches_not_modified(request.headers.get("if-none-match"), etag) or (
            request.headers.get("if-none-match") is None
            and _matches_since(request.headers.get("if-modified-since"), modified)
        ):
            handle.close()
            return Response(status_code=304, headers=common_headers)

        selected_range = None
        range_header = request.headers.get("range")
        if range_header and _if_range_matches(
            request.headers.get("if-range"), etag, modified
        ):
            try:
                selected_range = parse_byte_range(range_header, size)
            except RangeNotSatisfiable:
                handle.close()
                return JSONResponse(
                    status_code=416,
                    content={"detail": "requested byte range is not satisfiable"},
                    headers={
                        **common_headers,
                        "Content-Range": f"bytes */{size}",
                    },
                )

        start = 0 if selected_range is None else selected_range.start
        length = size if selected_range is None else selected_range.length
        response_status = 200 if selected_range is None else 206
        headers = {**common_headers, "Content-Length": str(length)}
        if selected_range is not None:
            headers["Content-Range"] = (
                f"bytes {selected_range.start}-{selected_range.end}/{size}"
            )

        return _VideoStreamResponse(
            handle,
            start=start,
            length=length,
            status_code=response_status,
            headers=headers,
            media_type=entry.media_type,
            head_only=request.method == "HEAD",
        )

    return router
