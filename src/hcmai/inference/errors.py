"""Inference error taxonomy for transport and capability failure classification.

Enforces typed failure boundaries so callers can distinguish between transient
unavailability, authentication failure, and malformed provider responses.
"""

from __future__ import annotations


class InferenceError(RuntimeError):
    """Base class for all inference-related failures."""


class InferenceUnavailableError(InferenceError):
    """Network failure, connect error, or remote timeout."""


class InferenceAuthError(InferenceError):
    """Authentication or authorization failure (HTTP 401/403)."""


class InferenceResponseError(InferenceError):
    """Malformed provider response, schema validation error, or non-2xx HTTP status."""


__all__ = [
    "InferenceError",
    "InferenceUnavailableError",
    "InferenceAuthError",
    "InferenceResponseError",
]
