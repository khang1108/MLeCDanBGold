"""Validated settings for the retrieval HTTP service and client.

This module parses environment settings for target host, deadlines, and
payload size limits for standalone retrieval serving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

DEFAULT_TARGET = "127.0.0.1:8002"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_HEALTH_TIMEOUT_SECONDS = 1.0
DEFAULT_MAX_MESSAGE_MIB = 64


@dataclass(frozen=True, slots=True)
class RetrievalClientSettings:
    """Client configuration for HTTP retrieval operations."""

    target: str = DEFAULT_TARGET
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_MIB * 1024 * 1024

    def __post_init__(self) -> None:
        """Validate client parameters to prevent malformed transport configurations."""
        if not self.target or not self.target.strip():
            raise ValueError("target must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive: {self.timeout_seconds}"
            )
        if self.health_timeout_seconds <= 0:
            raise ValueError(
                f"health_timeout_seconds must be positive: {self.health_timeout_seconds}"
            )
        min_bytes = 1 * 1024 * 1024
        max_bytes = 512 * 1024 * 1024
        if not (min_bytes <= self.max_message_bytes <= max_bytes):
            raise ValueError(
                f"max_message_bytes must be between 1 MiB and 512 MiB: {self.max_message_bytes}"
            )

    @property
    def base_url(self) -> str:
        """Derive standard HTTP base URL from target."""
        clean = self.target.strip()
        if clean.startswith("http://") or clean.startswith("https://"):
            return clean.rstrip("/")
        return f"http://{clean}"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RetrievalClientSettings:
        """Construct validated settings from an explicit mapping or os.environ."""
        source = os.environ if env is None else env

        target = source.get("HCMAI_RETRIEVAL_TARGET", DEFAULT_TARGET)

        timeout_str = source.get("HCMAI_RETRIEVAL_TIMEOUT_SECONDS")
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        if timeout_str is not None:
            try:
                timeout_seconds = float(timeout_str)
            except ValueError as error:
                raise ValueError(
                    f"HCMAI_RETRIEVAL_TIMEOUT_SECONDS must be numeric: {timeout_str}"
                ) from error

        health_str = source.get("HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS")
        health_timeout_seconds = DEFAULT_HEALTH_TIMEOUT_SECONDS
        if health_str is not None:
            try:
                health_timeout_seconds = float(health_str)
            except ValueError as error:
                raise ValueError(
                    f"HCMAI_RETRIEVAL_HEALTH_TIMEOUT_SECONDS must be numeric: {health_str}"
                ) from error

        max_mib_str = source.get("HCMAI_RETRIEVAL_MAX_MESSAGE_MIB")
        max_message_bytes = DEFAULT_MAX_MESSAGE_MIB * 1024 * 1024
        if max_mib_str is not None:
            try:
                max_message_bytes = int(max_mib_str) * 1024 * 1024
            except ValueError as error:
                raise ValueError(
                    f"HCMAI_RETRIEVAL_MAX_MESSAGE_MIB must be an integer: {max_mib_str}"
                ) from error

        return cls(
            target=target,
            timeout_seconds=timeout_seconds,
            health_timeout_seconds=health_timeout_seconds,
            max_message_bytes=max_message_bytes,
        )
