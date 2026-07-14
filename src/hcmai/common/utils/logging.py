"""Logging helpers for the HCMAI application and offline pipelines."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path


PathValue = str | PathLike[str]

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger without changing global logging configuration."""

    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: PathValue | None = None,
    log_format: str = DEFAULT_FORMAT,
    force: bool = False,
) -> None:
    """Configure console logging and optionally a UTF-8 log file.

    Args:
        level: Logging threshold, such as ``logging.INFO`` or ``"DEBUG"``.
        log_file: Optional path for a second file handler. Parent directories
            are created automatically.
        log_format: Format applied to all configured handlers.
        force: Remove existing root handlers before configuring logging.

    This function is intentionally explicit and should be called by an
    application entry point or command-line script, not during module import.
    """

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        output_path = Path(log_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(output_path, encoding="utf-8")
        )

    formatter = logging.Formatter(log_format)
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=force,
    )


__all__ = ["DEFAULT_FORMAT", "configure_logging", "get_logger"]
