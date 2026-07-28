"""Single-process ownership of hosted embedding, reranking, and LLM models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from hcmai.common.schemas import InferenceReadiness, ModelStatus
from hcmai.llm.config import LLMServiceConfig
from hcmai.llm.conversation import StructuredConversationModel
from hcmai.reranking.config import QwenRerankerConfig
from hcmai.reranking.qwen import QwenRerankerScorer
from hcmai.retriever.encoder import DenseEncoder


class LLMRuntime:
    """Load each configured model once and expose bounded inference methods."""

    def __init__(
        self,
        config: LLMServiceConfig,
        encoder: Any | None = None,
        reranker: Any | None = None,
        conversation: Any | None = None,
    ) -> None:
        self.config = config
        self.encoder = encoder or DenseEncoder(config.embedding)
        self.reranker = reranker or QwenRerankerScorer(_reranker_config(config))
        self.conversation = conversation or StructuredConversationModel(
            config.conversation
        )

    @classmethod
    def from_environment(cls) -> LLMRuntime:
        path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
        config = LLMServiceConfig.from_yaml(path)
        checkpoint = os.getenv("HCMAI_CONVERSATION_MODEL")
        if checkpoint:
            conversation = config.conversation.model_copy(
                update={"checkpoint": checkpoint}
            )
            config = config.model_copy(update={"conversation": conversation})
        return cls(config)

    def load(self) -> None:
        """Warm configured models during service lifespan, never at import."""
        self.encoder._load_model()
        self.reranker._ensure_loaded()
        self.conversation.load()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode_text(texts)

    def rerank(self, query: str, images: Sequence[Image.Image]) -> list[float]:
        return self.reranker.score_batch(query, images)

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.conversation(request)

    def readiness(self) -> InferenceReadiness:
        embedding_loaded = self.encoder.model is not None
        reranker_loaded = self.reranker._base_model is not None
        conversation_loaded = self.conversation.model is not None
        conversation_required = self.config.conversation.checkpoint is not None
        return InferenceReadiness(
            ready=embedding_loaded
            and reranker_loaded
            and (conversation_loaded or not conversation_required),
            models={
                "embedding": ModelStatus(
                    loaded=embedding_loaded,
                    checkpoint=self.config.embedding.model_name,
                ),
                "reranker": ModelStatus(
                    loaded=reranker_loaded,
                    checkpoint=self.config.reranker.checkpoint,
                    revision=self.reranker.resolved_revision,
                ),
                "conversation": ModelStatus(
                    loaded=conversation_loaded,
                    checkpoint=self.config.conversation.checkpoint,
                    revision=self.conversation.revision,
                ),
            },
        )


def _reranker_config(config: LLMServiceConfig) -> QwenRerankerConfig:
    values = config.reranker
    return QwenRerankerConfig(
        checkpoint=values.checkpoint,
        revision=values.revision,
        device=values.device,
        dtype=values.dtype,
        batch_size=values.batch_size,
        max_length=values.max_length,
        max_pixels=values.max_pixels,
    )
