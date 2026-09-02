"""Command-line entry point for the HCMAI local video origin."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Sequence

from .catalog import CatalogError, VideoCatalog
from .config import ConfigurationError, parse_settings
from .http_server import VideoHTTPServer

LOGGER = logging.getLogger("socketapp")


def _configure_logging(level: str) -> None:
    """Configure concise process logs without leaking request query strings."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ConfigurationError(f"invalid SOCKETAPP_LOG_LEVEL: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _install_shutdown_handlers(server: VideoHTTPServer) -> None:
    """Make SIGINT/SIGTERM stop ``serve_forever`` without a shutdown deadlock."""

    def request_shutdown(signum: int, _frame: object) -> None:
        LOGGER.info("shutdown requested signal=%s", signum)
        # BaseServer.shutdown must be called from a thread other than the one
        # currently running serve_forever; the helper avoids a signal deadlock.
        threading.Thread(
            target=server.shutdown,
            name="socketapp-shutdown",
            daemon=True,
        ).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)


def main(argv: Sequence[str] | None = None) -> int:
    """Load the local catalog, serve HTTP, and return a process exit code."""

    try:
        settings = parse_settings(None if argv is None else list(argv))
        _configure_logging(settings.log_level)
        catalog = VideoCatalog(
            settings.video_root,
            settings.manifest,
            allow_empty=settings.allow_empty,
        )
    except (ConfigurationError, CatalogError) as error:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.error("startup failed: %s", error)
        return 2

    try:
        server = VideoHTTPServer(settings, catalog)
    except OSError as error:
        LOGGER.error("could not bind %s:%d: %s", settings.host, settings.port, error)
        return 2
    _install_shutdown_handlers(server)
    LOGGER.info(
        "serving videos=%d bind=%s:%d workers=%d root=%s manifest=%s",
        len(catalog),
        settings.host,
        settings.port,
        settings.max_workers,
        catalog.root,
        catalog.manifest or "discovery",
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested from keyboard")
    finally:
        server.server_close()
        LOGGER.info("server stopped")
    return 0
