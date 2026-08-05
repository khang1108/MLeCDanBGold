"""Concrete local and HTTP LLM adapters."""

from hcmai.llm.adapters.http import InferenceClient, InferenceClientError
from hcmai.llm.adapters.local import LocalAdapter

__all__ = ["InferenceClient", "InferenceClientError", "LocalAdapter"]
