"""Environment configuration for the integrated local video origin.

Video routes are always registered, but catalog loading is opt-in through
``SOCKETAPP_VIDEO_ROOT`` so a backend without local source videos still starts.
Cloudflare and server bind settings remain outside this package because the
main HCMAI process owns its transport lifecycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class VideoConfigurationError(ValueError):
    """Raised when an explicitly configured video-origin value is invalid."""


@dataclass(frozen=True, slots=True)
class VideoOriginSettings:
    """Settings required to build the process-owned source-video catalog."""

    video_root: Path
    manifest: Path | None = None
    allow_empty: bool = False
    cache_control: str = "public, max-age=3600"


def _optional_environment(name: str) -> str | None:
    """Return one trimmed environment value, treating blanks as unset."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _boolean_environment(name: str, default: bool = False) -> bool:
    """Parse one explicit boolean without silently accepting spelling errors."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise VideoConfigurationError(f"{name} must be true or false")
    return normalized == "true"


def video_settings_from_environment() -> VideoOriginSettings | None:
    """Load catalog settings, or return ``None`` when video serving is disabled."""

    root = _optional_environment("SOCKETAPP_VIDEO_ROOT")
    if root is None:
        return None

    manifest = _optional_environment("SOCKETAPP_MANIFEST")
    return VideoOriginSettings(
        video_root=Path(root).expanduser(),
        manifest=None if manifest is None else Path(manifest).expanduser(),
        allow_empty=_boolean_environment("SOCKETAPP_ALLOW_EMPTY"),
        cache_control=os.getenv(
            "SOCKETAPP_CACHE_CONTROL", "public, max-age=3600"
        ).strip()
        or "public, max-age=3600",
    )
