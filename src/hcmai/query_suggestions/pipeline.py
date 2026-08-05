"""Public service facade for operator-requested query suggestions."""

from __future__ import annotations

import os
from collections.abc import Callable
from uuid import uuid4

import httpx

from hcmai.common.schemas import (
    QuerySuggestionInferenceRequest,
    QuerySuggestionRequest,
    QuerySuggestionResponse,
)
from hcmai.llm.pipeline import (
    LLMService,
    QuerySuggestionConfig,
    QuerySuggestionGenerationConfig,
)
from hcmai.query_suggestions.adapters.gpu import GPUClient, GPUSuggestionAdapter
from hcmai.query_suggestions.adapters.openai import OpenAIAdapter
from hcmai.query_suggestions.models import SuggestionAdapter
from hcmai.query_suggestions.prompting import (
    INSTRUCTION,
    parse_suggestions,
    suggestion_messages,
)

__all__ = [
    "INSTRUCTION",
    "QuerySuggestionConfig",
    "QuerySuggestionGenerationConfig",
    "SuggestionService",
    "build_query_suggestion_service",
    "parse_suggestions",
    "suggestion_messages",
]


class SuggestionService:
    """Apply the YAML default count and call exactly one selected provider."""

    def __init__(
        self,
        provider: SuggestionAdapter,
        default_count: int,
        provider_name: str,
    ) -> None:
        self.provider = provider
        self.default_count = default_count
        self.provider_name = provider_name

    def suggest(self, request: QuerySuggestionRequest) -> QuerySuggestionResponse:
        return self.provider.suggest(QuerySuggestionInferenceRequest(
            request_id=f"suggest-{uuid4().hex[:12]}",
            query=request.query,
            count=request.count or self.default_count,
        ))

    def close(self) -> None:
        """Release an owned provider client when it has one."""

        self.provider.close()


def build_query_suggestion_service(
    config: QuerySuggestionConfig,
    inference_client: LLMService | GPUClient | None,
    http_client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> SuggestionService | None:
    if not config.enabled:
        return None
    if config.active_provider == "gpu_inference":
        if inference_client is None:
            raise ValueError("GPU query suggestions require inference.enabled")
        values = config.gpu_inference

        provider: SuggestionAdapter = GPUSuggestionAdapter(
            inference_client, values.endpoint_path, values.timeout_seconds
        )
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
        provider = OpenAIAdapter(
            client, values.model, config.generation
        )
    return SuggestionService(
        provider, config.default_count, config.active_provider
    )
