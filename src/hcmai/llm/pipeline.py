"""Public lifecycle facade for local and remote LLM inference."""

from __future__ import annotations

from typing import Any

from hcmai.common.config import InferenceConfig
from hcmai.common.schemas import (
    InferenceCapabilities,
    InferenceReadiness,
    VQAInferenceEvidence,
)
from hcmai.llm.config import (
    HostedVQAConfig,
    LLMServiceConfig,
)

__all__ = [
    "HostedVQAConfig",
    "LLMService",
    "LLMServiceConfig",
]


class LLMService:
    """Expose one configured LLM deployment through a stable service boundary."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._last_readiness: InferenceReadiness | None = None

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

    def load(self) -> None:
        method = getattr(self.adapter, "load", None)
        if method is not None:
            method()

    def close(self) -> None:
        method = getattr(self.adapter, "close", None)
        if method is not None:
            method()

    def readiness(self, deadline_at: float | None = None) -> InferenceReadiness:
        self._last_readiness = self.adapter.readiness(deadline_at=deadline_at)
        return self._last_readiness

    def capability_health(self) -> dict[str, bool]:
        readiness = self._last_readiness
        capabilities = (
            InferenceCapabilities() if readiness is None else readiness.capabilities
        )
        return capabilities.model_dump()

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

    def answer_vqa(
        self,
        *,
        request_id: str,
        frame_id: str,
        question: str,
        image: Any,
        evidence: VQAInferenceEvidence,
    ) -> Any:
        return self.adapter.answer_vqa(
            request_id=request_id,
            frame_id=frame_id,
            question=question,
            image=image,
            evidence=evidence,
        )

    def answer_vqa_multi(
        self,
        *,
        request_id: str,
        frame_ids: list[str],
        question: str,
        images: list[Any],
        evidence: VQAInferenceEvidence,
    ) -> Any:
        method = getattr(self.adapter, "answer_vqa_multi", None)
        if method is None:
            raise RuntimeError("multi-frame VQA is not supported by this provider")
        return method(
            request_id=request_id,
            frame_ids=frame_ids,
            question=question,
            images=images,
            evidence=evidence,
        )
