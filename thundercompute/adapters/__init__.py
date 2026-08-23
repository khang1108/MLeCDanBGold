"""Concrete local and HTTP LLM adapters."""

from thundercompute.adapters.http import InferenceClient, InferenceClientError
from thundercompute.adapters.local import LocalAdapter
from thundercompute.adapters.pool import InferenceClientPool

__all__ = [
    "InferenceClient",
    "InferenceClientError",
    "InferenceClientPool",
    "LocalAdapter",
]
