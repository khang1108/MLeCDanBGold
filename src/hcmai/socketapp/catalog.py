"""Safe, immutable-at-runtime catalog of local HCMAI video files.

The catalog owns video-ID-to-file resolution and path validation. It does not
serve bytes, infer metadata from HCMAI frames, or fall back to YouTube. A JSON
manifest is preferred when canonical IDs differ from local filenames; recursive
discovery is provided for the common ``<video_id>.<extension>`` layout.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import shutil
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

VIDEO_MEDIA_TYPES: dict[str, str] = {
    ".avi": "video/x-msvideo",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
FFPROBE_TIMEOUT_SECONDS = 5


class CatalogError(ValueError):
    """Raised when the local video catalog cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class VideoEntry:
    """One validated video identity and its resolved local path."""

    video_id: str
    path: Path
    media_type: str
    fps: float | None = None

    def snapshot(self) -> os.stat_result:
        """Return the current file metadata immediately before a transfer."""

        stat_result = self.path.stat()
        if not stat.S_ISREG(stat_result.st_mode):
            raise FileNotFoundError(str(self.path))
        return stat_result


def _is_supported_video(path: Path) -> bool:
    """Return whether a path has a media extension the service understands."""

    return path.suffix.lower() in VIDEO_MEDIA_TYPES


def _validate_video_id(video_id: object) -> str:
    """Validate one URL-addressable canonical video ID."""

    if not isinstance(video_id, str):
        raise CatalogError("video_id must be a string")
    value = video_id.strip()
    if not value or value in {".", ".."}:
        raise CatalogError("video_id must be non-empty")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise CatalogError(
            f"video_id {value!r} cannot contain slash, backslash, or NUL"
        )
    return value


def _resolve_inside_root(root: Path, raw_path: object) -> Path:
    """Resolve a manifest path and reject traversal outside ``root``."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        raise CatalogError("manifest video path must be a non-empty string")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError) as error:
        raise CatalogError(f"video path does not exist: {raw_path!r}") from error
    except ValueError as error:
        raise CatalogError(
            f"video path escapes SOCKETAPP_VIDEO_ROOT: {raw_path!r}"
        ) from error
    if not resolved.is_file():
        raise CatalogError(f"video path is not a regular file: {raw_path!r}")
    if not _is_supported_video(resolved):
        raise CatalogError(
            f"unsupported video extension for {raw_path!r}; "
            f"supported: {', '.join(sorted(VIDEO_MEDIA_TYPES))}"
        )
    return resolved


def _media_type(path: Path, record: Mapping[str, Any] | None = None) -> str:
    """Choose a safe media type from a manifest override or file extension."""

    if record is not None:
        declared = record.get("mime_type", record.get("media_type"))
        if isinstance(declared, str) and declared.startswith("video/"):
            return declared
    return VIDEO_MEDIA_TYPES.get(
        path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    )


def _parse_fps(value: object) -> float | None:
    """Parse a positive frame rate from a number or an FFmpeg fraction."""

    try:
        if isinstance(value, str) and "/" in value:
            parsed = float(Fraction(value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            parsed = float(value)
        else:
            return None
    except (ArithmeticError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 1_000:
        return None
    return parsed


def _declared_fps(record: Mapping[str, Any] | None) -> float | None:
    """Read an optional manifest FPS override before probing the media file."""

    if record is None:
        return None
    for key in ("fps", "frame_rate", "frame_rate_fps"):
        if key in record:
            return _parse_fps(record[key])
    return None


def _probe_fps(path: Path) -> float | None:
    """Read the source video FPS once with optional local ``ffprobe`` support."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        stream = payload.get("streams", [])[0]
    except (IndexError, AttributeError, TypeError, json.JSONDecodeError):
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        fps = _parse_fps(stream.get(key))
        if fps is not None:
            return fps
    return None


def _fps(path: Path, record: Mapping[str, Any] | None = None) -> float | None:
    """Choose a manifest FPS override or probe the source file once."""

    return _declared_fps(record) or _probe_fps(path)


def _record_parts(record: Mapping[str, Any]) -> tuple[str, object, Mapping[str, Any]]:
    """Extract a video ID and path from one supported manifest record."""

    video_id = record.get("video_id", record.get("id"))
    raw_path = record.get("path", record.get("file", record.get("filename")))
    if video_id is None or raw_path is None:
        raise CatalogError("each manifest record requires video_id and path")
    return _validate_video_id(video_id), raw_path, record


def _manifest_records(payload: object) -> list[tuple[str, object, Mapping[str, Any]]]:
    """Normalize supported JSON manifest shapes into records."""

    if isinstance(payload, Mapping) and "videos" in payload:
        payload = payload["videos"]

    if isinstance(payload, list):
        records: list[tuple[str, object, Mapping[str, Any]]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                raise CatalogError("manifest videos list contains a non-object")
            records.append(_record_parts(item))
        return records

    if isinstance(payload, Mapping):
        records = []
        for video_id, item in payload.items():
            if isinstance(item, str):
                record: Mapping[str, Any] = {"video_id": video_id, "path": item}
            elif isinstance(item, Mapping):
                record = {"video_id": video_id, **item}
            else:
                raise CatalogError(
                    f"manifest value for {video_id!r} must be a path or object"
                )
            records.append(_record_parts(record))
        return records

    raise CatalogError("manifest must be a list or JSON object")


class VideoCatalog:
    """In-memory index of validated local videos for O(1) request lookup."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path | None = None,
        *,
        allow_empty: bool = False,
    ) -> None:
        """Load a manifest or discover videos below ``root`` once at startup.

        Args:
            root: Directory that every served file must remain inside.
            manifest: Optional JSON manifest. Relative paths in the manifest
                are relative to the catalog root.
            allow_empty: Permit startup with zero videos for health/debug use.

        Raises:
            CatalogError: If the root, manifest, IDs, or file paths are invalid.
        """

        try:
            self.root = Path(root).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError) as error:
            raise CatalogError(f"video root does not exist: {root}") from error
        if not self.root.is_dir():
            raise CatalogError(f"video root is not a directory: {self.root}")

        self.manifest = None if manifest is None else Path(manifest).expanduser()
        if self.manifest is not None and not self.manifest.is_absolute():
            self.manifest = Path.cwd() / self.manifest
        if self.manifest is not None:
            self.manifest = self.manifest.resolve()

        try:
            entries = (
                self._load_manifest(self.manifest)
                if self.manifest is not None
                else self._discover()
            )
        except OSError as error:
            source = self.manifest or self.root
            raise CatalogError(f"could not scan video catalog: {source}") from error
        if not entries and not allow_empty:
            source = self.manifest or self.root
            raise CatalogError(f"no supported videos found in {source}")
        self._entries = entries

    def _load_manifest(self, manifest: Path) -> dict[str, VideoEntry]:
        """Load and validate an explicit JSON catalog."""

        if not manifest.is_file():
            raise CatalogError(f"manifest does not exist: {manifest}")
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CatalogError(f"could not read manifest: {manifest}") from error

        entries: dict[str, VideoEntry] = {}
        for video_id, raw_path, record in _manifest_records(payload):
            if video_id in entries:
                raise CatalogError(f"duplicate video_id in manifest: {video_id}")
            path = _resolve_inside_root(self.root, raw_path)
            entries[video_id] = VideoEntry(
                video_id=video_id,
                path=path,
                media_type=_media_type(path, record),
                fps=_fps(path, record),
            )
        return entries

    def _discover(self) -> dict[str, VideoEntry]:
        """Discover supported files and use each filename stem as its ID."""

        entries: dict[str, VideoEntry] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or not _is_supported_video(path):
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(self.root)
            except ValueError as error:
                raise CatalogError(
                    f"discovered video escapes video root: {path}"
                ) from error
            video_id = _validate_video_id(path.stem)
            if video_id in entries:
                raise CatalogError(
                    f"duplicate discovered video_id {video_id!r}; "
                    "provide SOCKETAPP_MANIFEST to disambiguate files"
                )
            entries[video_id] = VideoEntry(
                video_id=video_id,
                path=resolved,
                media_type=_media_type(resolved),
                fps=_fps(resolved),
            )
        return entries

    def get(self, video_id: str) -> VideoEntry | None:
        """Return one exact canonical video entry, or ``None`` if unknown."""

        return self._entries.get(video_id)

    def __contains__(self, video_id: str) -> bool:
        """Return whether an exact video ID is present."""

        return video_id in self._entries

    def __len__(self) -> int:
        """Return the number of catalog entries."""

        return len(self._entries)

    def entries(self) -> tuple[VideoEntry, ...]:
        """Return entries in deterministic video-ID order."""

        return tuple(self._entries[key] for key in sorted(self._entries))
