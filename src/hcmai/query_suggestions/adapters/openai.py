"""OpenAI-compatible suggestion adapter."""

from __future__ import annotations

import json
from time import perf_counter

import httpx

from hcmai.common.schemas import (
    QuerySuggestionInferenceRequest,
    QuerySuggestionResponse,
)
from hcmai.llm.pipeline import QuerySuggestionGenerationConfig
from hcmai.query_suggestions.prompting import INSTRUCTION, parse_suggestions


class OpenAIAdapter:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        generation: QuerySuggestionGenerationConfig,
    ) -> None:
        self.client = client
        self.model = model
        self.generation = generation

    def suggest(
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
