"""Concrete local and HTTP LLM adapters."""

from hcmai.thundercompute.adapters.http import InferenceClient, InferenceClientError
from hcmai.thundercompute.adapters.local import LocalAdapter
from hcmai.thundercompute.adapters.pool import InferenceClientPool

__all__ = [
    "InferenceClient",
    "InferenceClientError",
    "InferenceClientPool",
    "LocalAdapter",
]
