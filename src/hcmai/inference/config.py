"""Environment-backed configuration for provider-agnostic inference endpoints.

This module provides typed endpoint settings for LLM and embedding capabilities.
It reads configuration from standard environment variables (HCMAI_LLM_* and
HCMAI_EMBEDDING_*) without embedding provider-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True, slots=True)
class ModelEndpointConfig:
    """Connection parameters and model identity for a remote inference provider."""

    base_url: str
    api_key: str | None = None
    model: str = ""
    timeout_seconds: float = 30.0


def _load(prefix: str, *, default_timeout: float = 30.0) -> ModelEndpointConfig:
    """Load and validate an endpoint configuration from environment variables."""
    base_url_key = f"HCMAI_{prefix}_BASE_URL"
    if base_url_key not in os.environ:
        raise KeyError(f"Missing required environment variable: {base_url_key}")
    base_url = os.environ[base_url_key].rstrip("/")

    model_key = f"HCMAI_{prefix}_MODEL"
    if model_key not in os.environ:
        raise KeyError(f"Missing required environment variable: {model_key}")
    model = os.environ[model_key].strip()
    if not model:
        raise ValueError(f"{model_key} must not be blank")

    api_key = os.getenv(f"HCMAI_{prefix}_API_KEY") or None

    timeout_str = os.getenv(f"HCMAI_{prefix}_TIMEOUT_SECONDS", str(default_timeout))
    try:
        timeout = float(timeout_str)
    except ValueError as exc:
        raise ValueError(f"HCMAI_{prefix}_TIMEOUT_SECONDS must be a valid float") from exc

    if timeout <= 0:
        raise ValueError(f"HCMAI_{prefix}_TIMEOUT_SECONDS must be positive")

    return ModelEndpointConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout,
    )


def load_llm_endpoint() -> ModelEndpointConfig:
    """Load configuration for the LLM structured text generation provider."""
    return _load("LLM")


def load_embedding_endpoint() -> ModelEndpointConfig:
    """Load configuration for the text embedding provider."""
    return _load("EMBEDDING")
