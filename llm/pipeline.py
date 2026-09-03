"""Public lifecycle facade for local and remote LLM inference."""

from __future__ import annotations

from typing import Any

from hcmai.common.config import InferenceConfig
from llm.config import LLMServiceConfig

__all__ = [
    "LLMService",
    "LLMServiceConfig",
]


class LLMService:
    """Expose one configured LLM deployment through a stable service boundary."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._last_readiness: Any | None = None

    @classmethod
    def from_environment(cls) -> LLMService:
        from llm.local import LocalAdapter

        return cls(LocalAdapter.from_environment())

    @classmethod
    def remote(
        cls,
        base_url: str,
        timeout_seconds: float | InferenceConfig = 10,
        client: Any | None = None,
    ) -> LLMService:
        from llm.remote.client import InferenceClient

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
    def query_preparer(self) -> Any:
        """Return the hosted query-preparation adapter when available."""

        return self.adapter.query_preparer

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

    def readiness(self, *args: Any, **kwargs: Any) -> Any:
        self._last_readiness = self.adapter.readiness(*args, **kwargs)
        return self._last_readiness

    def capability_health(self) -> dict[str, bool]:
        readiness = self._last_readiness
        if readiness is None:
            return {
                "embedding": False,
                "reranking": False,
                "structured_parsing": False,
                "query_preparation": False,
            }
        return readiness.capabilities.model_dump()

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

    def embed_images(
        self,
        images: Any,
        *,
        source: str = "visual",
        item_ids: list[str] | None = None,
    ) -> Any:
        """Embed decoded images using the requested visual encoder."""

        method = getattr(self.adapter, "embed_images", None)
        if method is None:
            raise RuntimeError("image embedding is not supported by this provider")
        if item_ids is None:
            return method(images, source=source)
        return method(images, source=source, item_ids=item_ids)

    def caption(self, images: Any) -> Any:
        return self.adapter.caption(images)

    def ocr(self, images: Any) -> Any:
        """Run structured OCR through the configured inference adapter."""

        return self.adapter.ocr(images)

    def rerank(self, query: str, images: Any) -> list[float]:
        return self.adapter.rerank(query, images)

    def translate_query_events(self, events: list[str]) -> list[str]:
        """Translate ordered retrieval events through the configured provider."""

        return self.adapter.translate_query_events(events)

    def generate_query_candidates(
        self, events: list[str], candidate_count: int = 5
    ) -> dict[str, Any]:
        """Generate literal and controlled retrieval-event bundles."""

        return self.adapter.generate_query_candidates(events, candidate_count)

    def boundary_scores(self, frames: Any, *, source: str) -> Any:
        return self.adapter.boundary_scores(frames, source=source)

    def transcribe_reference(self, payload: Any) -> Any:
        return self.adapter.transcribe_reference(payload)

    def diarize_reference(self, payload: Any) -> Any:
        return self.adapter.diarize_reference(payload)

    def embed_dino(self, images: Any) -> Any:
        method = getattr(self.adapter, "embed_images", None)
        if method is not None:
            return method(images, source="dino")
        raise RuntimeError("dino embedding is not supported by this provider")
