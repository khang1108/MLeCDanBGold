"""Logging helpers for the HCMAI application and offline pipelines."""

from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path

PathValue = str | PathLike[str]

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d (%(funcName)s) | %(message)s"
)

_RESET = "\033[0m"
_DIM = "\033[2m"
_NAME_COLOR = "\033[36m"
_LOCATION_COLOR = "\033[35m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class ColoredFormatter(logging.Formatter):
    """Colorize timestamp, level, logger name, and file:line on the console."""

    def formatTime(
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Dim the timestamp so status and location stand out."""
        return f"{_DIM}{logging.Formatter.formatTime(self, record, datefmt)}{_RESET}"

    def format(self, record: logging.LogRecord) -> str:
        """Format a record while restoring fields mutated for terminal colors."""
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        original = (record.levelname, record.name, record.filename, record.funcName)
        record.levelname = f"{level_color}{record.levelname}{_RESET}"
        record.name = f"{_NAME_COLOR}{record.name}{_RESET}"
        record.filename = f"{_LOCATION_COLOR}{record.filename}"
        record.funcName = f"{record.funcName}{_RESET}"
        try:
            return logging.Formatter.format(self, record)
        finally:
            record.levelname, record.name, record.filename, record.funcName = original


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
