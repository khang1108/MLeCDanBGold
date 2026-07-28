"""Logging helpers for the HCMAI application and offline pipelines."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path


PathValue = str | PathLike[str]

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "[%(pathname)s, %(filename)s/%(funcName)s:%(lineno)d] | %(message)s"
)

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class ColoredFormatter(logging.Formatter):
    """Add an ANSI color to the level name without altering the log record."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return super().format(record)

        original_levelname = record.levelname
        record.levelname = f"{color}{original_levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


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

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(log_format))
    handlers: list[logging.Handler] = [console_handler]

    if log_file is not None:
        output_path = Path(log_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=force,
    )


__all__ = [
    "DEFAULT_FORMAT",
    "ColoredFormatter",
    "configure_logging",
    "get_logger",
]
