"""Public lifecycle facade for local and remote LLM inference."""

from __future__ import annotations

from typing import Any

from hcmai.common.config import InferenceConfig
from hcmai.llm.config import (
    HostedConversationConfig,
    LLMServiceConfig,
    QuerySuggestionConfig,
    QuerySuggestionGenerationConfig,
)

__all__ = [
    "HostedConversationConfig",
    "LLMService",
    "LLMServiceConfig",
    "QuerySuggestionConfig",
    "QuerySuggestionGenerationConfig",
]


class LLMService:
    """Expose one configured LLM deployment through a stable service boundary."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    @classmethod
    def from_environment(cls) -> LLMService:
        from hcmai.llm.adapters.local import LocalAdapter

        return cls(LocalAdapter.from_environment())

    @classmethod
    def remote(
        cls,
        base_url: str,
        timeout_seconds: float | InferenceConfig = 10,
        client: Any | None = None,
    ) -> LLMService:
        from hcmai.llm.adapters.http import InferenceClient

        return cls(InferenceClient(base_url, timeout_seconds, client))

    @property
    def config(self) -> Any:
        return self.adapter.config

    @property
    def captioner(self) -> Any:
        return self.adapter.captioner

    @property
    def reranker(self) -> Any:
        return self.adapter.reranker

    @property
    def query_suggester(self) -> Any:
        return self.adapter.query_suggester

    def load(self) -> None:
        method = getattr(self.adapter, "load", None)
        if method is not None:
            method()

    def close(self) -> None:
        method = getattr(self.adapter, "close", None)
        if method is not None:
            method()
            return
        client = getattr(self.adapter, "client", None)
        method = getattr(client, "close", None)
        if method is not None:
            method()

    def readiness(self) -> Any:
        return self.adapter.readiness()

    def gateway_health(self) -> dict[str, Any]:
        method = getattr(self.adapter, "health", None)
        if method is None:
            return {
                "configured": False,
                "circuit_state": "not_applicable",
            }
        return method()

    def embed_text(self, texts: list[str], source: str = "visual") -> Any:
        return self.adapter.embed_text(texts, source)

    def caption(self, images: Any) -> Any:
        return self.adapter.caption(images)

    def rerank(self, query: str, images: Any) -> list[float]:
        return self.adapter.rerank(query, images)

    def resolve(self, request: dict[str, Any]) -> Any:
        method = getattr(self.adapter, "resolve", None)
        if method is None:
            method = self.adapter.resolve_conversation
        return method(request)

    def suggest_queries(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.suggest_queries(*args, **kwargs)

    def answer_vqa(self, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.answer_vqa(*args, **kwargs)
