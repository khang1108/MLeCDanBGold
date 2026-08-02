"""Configured providers for operator-requested query suggestions."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from time import perf_counter
from typing import Protocol
from uuid import uuid4

import httpx

from hcmai.common.schemas import (
    QuerySuggestionInferenceRequest,
    QuerySuggestionRequest,
    QuerySuggestionResponse,
)
from hcmai.llm.config import (
    QuerySuggestionConfig,
    QuerySuggestionGenerationConfig,
)
from hcmai.llm.models.query_suggestion import INSTRUCTION, parse_suggestions


class QuerySuggestionProvider(Protocol):
    def __call__(
        self, request: QuerySuggestionInferenceRequest
    ) -> QuerySuggestionResponse: ...


class GpuSuggestionClient(Protocol):
    def suggest_queries(
        self,
        request: QuerySuggestionInferenceRequest,
        endpoint_path: str,
        timeout_seconds: float,
    ) -> QuerySuggestionResponse: ...


class QuerySuggestionService:
    """Apply the YAML default count and call exactly one selected provider."""

    def __init__(
        self,
        provider: QuerySuggestionProvider,
        default_count: int,
        provider_name: str,
    ) -> None:
        self.provider = provider
        self.default_count = default_count
        self.provider_name = provider_name

    def suggest(self, request: QuerySuggestionRequest) -> QuerySuggestionResponse:
        return self.provider(QuerySuggestionInferenceRequest(
            request_id=f"suggest-{uuid4().hex[:12]}",
            query=request.query,
            count=request.count or self.default_count,
        ))


def build_query_suggestion_service(
    config: QuerySuggestionConfig,
    inference_client: GpuSuggestionClient | None,
    http_client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> QuerySuggestionService | None:
    if not config.enabled:
        return None
    if config.active_provider == "gpu_inference":
        if inference_client is None:
            raise ValueError("GPU query suggestions require inference.enabled")
        values = config.gpu_inference

        def gpu(request: QuerySuggestionInferenceRequest) -> QuerySuggestionResponse:
            return inference_client.suggest_queries(
                request, values.endpoint_path, values.timeout_seconds
            )

        provider: QuerySuggestionProvider = gpu
    else:
        values = config.openai_compatible
        api_key = os.getenv(values.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing query-suggestion API key in {values.api_key_env}"
            )
        client = http_client_factory(
            base_url=values.base_url.rstrip("/") + "/",
            timeout=values.timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        provider = _OpenAICompatibleProvider(
            client, values.model, config.generation
        )
    return QuerySuggestionService(
        provider, config.default_count, config.active_provider
    )


class _OpenAICompatibleProvider:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        generation: QuerySuggestionGenerationConfig,
    ) -> None:
        self.client = client
        self.model = model
        self.generation = generation

    def __call__(
        self, request: QuerySuggestionInferenceRequest
    ) -> QuerySuggestionResponse:
        started = perf_counter()
        response = self.client.post("chat/completions", json={
            "model": self.model,
            "max_tokens": self.generation.max_new_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "messages": [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": json.dumps({
                    "original_query": request.query,
                    "count": request.count,
                }, ensure_ascii=False)},
            ],
        })
        response.raise_for_status()
        payload = response.json()
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid OpenAI-compatible response") from error
        suggestions = parse_suggestions(str(text), request.query, request.count)
        return QuerySuggestionResponse(
            request_id=request.request_id,
            original_query=request.query,
            suggestions=suggestions,
            provider="openai_compatible",
            model=self.model,
            generation_latency_ms=(perf_counter() - started) * 1_000,
        )

    def close(self) -> None:
        self.client.close()
