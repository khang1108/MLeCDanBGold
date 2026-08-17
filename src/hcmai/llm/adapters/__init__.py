"""Concrete local and HTTP LLM adapters."""

from hcmai.llm.adapters.http import InferenceClient, InferenceClientError
from hcmai.llm.adapters.local import LocalAdapter
from hcmai.llm.adapters.pool import InferenceClientPool

__all__ = [
    "InferenceClient",
    "InferenceClientError",
    "InferenceClientPool",
    "LocalAdapter",
]
