"""Owned GPU-inference suggestion adapter."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import (
    QuerySuggestionInferenceRequest,
    QuerySuggestionResponse,
)


class GPUClient(Protocol):
    def suggest_queries(
        self,
        request: QuerySuggestionInferenceRequest,
        endpoint_path: str,
        timeout_seconds: float,
    ) -> QuerySuggestionResponse: ...


class GPUSuggestionAdapter:
    def __init__(
        self, client: GPUClient, endpoint_path: str, timeout_seconds: float
    ) -> None:
        self.client = client
        self.endpoint_path = endpoint_path
        self.timeout_seconds = timeout_seconds

    def suggest(
        self, request: QuerySuggestionInferenceRequest
    ) -> QuerySuggestionResponse:
        return self.client.suggest_queries(
            request, self.endpoint_path, self.timeout_seconds
        )

    def close(self) -> None:
        """The shared LLM service is closed by the composition root."""
