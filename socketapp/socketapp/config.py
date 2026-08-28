"""Environment and command-line configuration for the local video origin.

This module owns parsing and validation of runtime settings. It intentionally
does not load dotenv files or read Cloudflare secrets; process supervisors and
the tunnel runtime should provide those separately.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VIDEO_ROOT = (
    Path.home() / "personal" / "MLeCDanBGold" / "data" / "videos"
)
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://localhost:3000",
)


class ConfigurationError(ValueError):
    """Raised when a SocketApp setting cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for the HTTP server.

    ``video_root`` is the only data source. The default host is loopback so a
    Cloudflare Tunnel or reverse proxy must be deliberately configured before
    the service is reachable from outside the machine.
    """

    host: str = "127.0.0.1"
    port: int = 8120
    video_root: Path = DEFAULT_VIDEO_ROOT
    manifest: Path | None = None
    max_workers: int = 32
    request_timeout_seconds: float = 30.0
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    allow_empty: bool = False
    log_level: str = "INFO"
    cache_control: str = "public, max-age=3600"


def _read_env(name: str, default: str | None = None) -> str | None:
    """Read one environment value and treat blank strings as unset."""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _parse_positive_int(name: str, value: str, *, maximum: int) -> int:
    """Parse a bounded positive integer setting with a useful error."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if parsed <= 0 or parsed > maximum:
        raise ConfigurationError(
            f"{name} must be between 1 and {maximum}, got {parsed}"
        )
    return parsed


def _parse_timeout(value: str) -> float:
    """Parse a positive request timeout in seconds."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(
            "SOCKETAPP_REQUEST_TIMEOUT_SECONDS must be a number"
        ) from error
    if parsed <= 0 or parsed > 600:
        raise ConfigurationError(
            "SOCKETAPP_REQUEST_TIMEOUT_SECONDS must be between 0 and 600"
        )
    return parsed


def _parse_origins(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated browser origins, preserving explicit wildcard."""

    if value is None:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(item.strip() for item in value.split(",") if item.strip())
    if not origins:
        raise ConfigurationError("SOCKETAPP_CORS_ORIGINS cannot be blank")
    if "*" in origins and len(origins) != 1:
        raise ConfigurationError(
            "SOCKETAPP_CORS_ORIGINS cannot combine * with explicit origins"
        )
    return origins


def settings_from_environment() -> Settings:
    """Build settings from ``SOCKETAPP_*`` environment variables only."""

    root = Path(
        _read_env("SOCKETAPP_VIDEO_ROOT", str(DEFAULT_VIDEO_ROOT))
    ).expanduser()
    manifest_value = _read_env("SOCKETAPP_MANIFEST")
    manifest = Path(manifest_value).expanduser() if manifest_value else None

    port = _parse_positive_int(
        "SOCKETAPP_PORT", _read_env("SOCKETAPP_PORT", "8120"), maximum=65535
    )
    workers = _parse_positive_int(
        "SOCKETAPP_MAX_WORKERS",
        _read_env("SOCKETAPP_MAX_WORKERS", "32"),
        maximum=512,
    )
    timeout = _parse_timeout(
        _read_env("SOCKETAPP_REQUEST_TIMEOUT_SECONDS", "30")
    )
    allow_empty = _read_env("SOCKETAPP_ALLOW_EMPTY", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return Settings(
        host=_read_env("SOCKETAPP_HOST", "127.0.0.1"),
        port=port,
        video_root=root,
        manifest=manifest,
        max_workers=workers,
        request_timeout_seconds=timeout,
        cors_origins=_parse_origins(_read_env("SOCKETAPP_CORS_ORIGINS")),
        allow_empty=allow_empty,
        log_level=_read_env("SOCKETAPP_LOG_LEVEL", "INFO").upper(),
        cache_control=_read_env(
            "SOCKETAPP_CACHE_CONTROL", "public, max-age=3600"
        ),
    )


def parse_settings(argv: list[str] | None = None) -> Settings:
    """Parse CLI overrides on top of environment-backed settings.

    CLI options are intentionally limited to operational values. Credentials
    and tunnel configuration are not accepted because they do not belong to
    this process.
    """

    environment = settings_from_environment()
    parser = argparse.ArgumentParser(
        description="Serve local HCMAI videos with HTTP byte-range support."
    )
    parser.add_argument("--host", default=None, help="Bind address")
    parser.add_argument("--port", type=int, default=None, help="TCP port")
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        default=None,
        help="Allowed browser origin; repeat for multiple origins or use *",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Start even when the catalog contains no videos",
    )
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    port = environment.port if args.port is None else args.port
    if port <= 0 or port > 65535:
        raise ConfigurationError(f"--port must be between 1 and 65535, got {port}")
    workers = environment.max_workers if args.workers is None else args.workers
    if workers <= 0 or workers > 512:
        raise ConfigurationError(f"--workers must be between 1 and 512, got {workers}")
    timeout = (
        environment.request_timeout_seconds
        if args.request_timeout is None
        else args.request_timeout
    )
    if timeout <= 0 or timeout > 600:
        raise ConfigurationError(
            "--request-timeout must be between 0 and 600 seconds"
        )

    origins = environment.cors_origins
    if args.cors_origins is not None:
        origins = tuple(
            origin.strip()
            for value in args.cors_origins
            for origin in value.split(",")
            if origin.strip()
        )
        if not origins:
            raise ConfigurationError("--cors-origin must not be blank")
        if "*" in origins and len(origins) != 1:
            raise ConfigurationError(
                "--cors-origin cannot combine * with explicit origins"
            )

    manifest = environment.manifest if args.manifest is None else args.manifest
    return Settings(
        host=environment.host if args.host is None else args.host,
        port=port,
        video_root=(
            environment.video_root
            if args.video_root is None
            else args.video_root.expanduser()
        ),
        manifest=None if manifest is None else manifest.expanduser(),
        max_workers=workers,
        request_timeout_seconds=timeout,
        cors_origins=origins,
        allow_empty=environment.allow_empty or args.allow_empty,
        log_level=environment.log_level if args.log_level is None else args.log_level.upper(),
        cache_control=environment.cache_control,
    )
