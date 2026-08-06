"""Concrete query-suggestion provider adapters."""

from hcmai.query_suggestions.adapters.gpu import GPUSuggestionAdapter
from hcmai.query_suggestions.adapters.openai import OpenAIAdapter

__all__ = ["GPUSuggestionAdapter", "OpenAIAdapter"]
