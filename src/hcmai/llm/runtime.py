"""Single-process ownership of hosted embedding, reranking, and LLM models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from hcmai.common.schemas import InferenceReadiness, ModelStatus
from hcmai.enrichment.caption.backend import FrameCaptioner
from hcmai.llm.config import LLMServiceConfig
from hcmai.llm.conversation import StructuredConversationModel
from hcmai.reranking.qwen import QwenRerankerConfig, QwenRerankerScorer
from hcmai.retriever.dense import DenseEncoder


class LLMRuntime:
    """Load each configured model once and expose bounded inference methods."""

    def __init__(
        self,
        config: LLMServiceConfig,
        visual_encoder: Any | None = None,
        caption_encoder: Any | None = None,
        captioner: Any | None = None,
        reranker: Any | None = None,
        conversation: Any | None = None,
        *,
        enable_caption: bool = True,
        enable_embedding: bool = True,
        enable_reranker: bool = True,
        enable_conversation: bool = True,
    ) -> None:
        self.config = config
        self.enable_caption = enable_caption
        self.enable_embedding = enable_embedding
        self.enable_reranker = enable_reranker
        self.enable_conversation = enable_conversation
        self.visual_encoder = visual_encoder or (
            DenseEncoder(config.visual_embedding) if enable_embedding else None
        )
        self.caption_encoder = caption_encoder or (
            (
                self.visual_encoder
                if config.caption_embedding == config.visual_embedding
                else DenseEncoder(config.caption_embedding)
            )
            if enable_embedding
            else None
        )
        self.captioner = captioner or (
            FrameCaptioner(config.caption_generation) if enable_caption else None
        )
        self.reranker = reranker or (
            QwenRerankerScorer(_reranker_config(config))
            if enable_reranker
            else None
        )
        self.conversation = conversation or (
            StructuredConversationModel(config.conversation)
            if enable_conversation
            else None
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
        return cls(
            config,
            enable_caption=_env_bool("HCMAI_ENABLE_CAPTION"),
            enable_embedding=_env_bool("HCMAI_ENABLE_EMBEDDING"),
            enable_reranker=_env_bool("HCMAI_ENABLE_RERANKER"),
            enable_conversation=_env_bool("HCMAI_ENABLE_CONVERSATION"),
        )

    def load(self) -> None:
        """Warm configured models during service lifespan, never at import."""
        if self.captioner is not None:
            self.captioner.resolve_revision()
        if self.visual_encoder is not None:
            self.visual_encoder._load_model()
        if (
            self.caption_encoder is not None
            and self.caption_encoder is not self.visual_encoder
        ):
            self.caption_encoder._load_model()
        if self.reranker is not None:
            self.reranker._ensure_loaded()
        if self.conversation is not None:
            self.conversation.load()

    def embed_text(self, texts: list[str], source: str = "visual") -> np.ndarray:
        encoder = (
            self.caption_encoder if source == "caption" else self.visual_encoder
        )
        if encoder is None:
            raise RuntimeError("embedding model is disabled")
        return encoder.encode_text(texts)

    def caption(self, images: Sequence[Image.Image]) -> list[str]:
        if self.captioner is None:
            raise RuntimeError("caption model is disabled")
        results = self.captioner.caption_batch(images)
        if any(isinstance(value, Exception) for value in results):
            raise RuntimeError("caption model returned a per-image failure")
        return [str(value).strip() for value in results]

    def rerank(self, query: str, images: Sequence[Image.Image]) -> list[float]:
        if self.reranker is None:
            raise RuntimeError("reranker model is disabled")
        return self.reranker.score_batch(query, images)

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.conversation is None:
            raise RuntimeError("conversation model is disabled")
        return self.conversation(request)

    def readiness(self) -> InferenceReadiness:
        generator_loaded = (
            self.captioner is not None and self.captioner.model is not None
        )
        visual_loaded = (
            self.visual_encoder is not None and self.visual_encoder.model is not None
        )
        caption_loaded = (
            self.caption_encoder is not None and self.caption_encoder.model is not None
        )
        reranker_loaded = (
            self.reranker is not None and self.reranker._base_model is not None
        )
        conversation_loaded = (
            self.conversation is not None and self.conversation.model is not None
        )
        return InferenceReadiness(
            ready=(not self.enable_caption or generator_loaded)
            and (not self.enable_embedding or visual_loaded and caption_loaded)
            and (not self.enable_reranker or reranker_loaded)
            and (not self.enable_conversation or conversation_loaded),
            models={
                "caption_generation": ModelStatus(
                    enabled=self.enable_caption,
                    loaded=generator_loaded,
                    checkpoint=self.config.caption_generation.model_checkpoint,
                    revision=(
                        self.captioner.resolved_revision
                        if self.captioner is not None
                        else None
                    ),
                ),
                "visual_embedding": ModelStatus(
                    enabled=self.enable_embedding,
                    loaded=visual_loaded,
                    checkpoint=self.config.visual_embedding.model_name,
                ),
                "caption_embedding": ModelStatus(
                    enabled=self.enable_embedding,
                    loaded=caption_loaded,
                    checkpoint=self.config.caption_embedding.model_name,
                ),
                "reranker": ModelStatus(
                    enabled=self.enable_reranker,
                    loaded=reranker_loaded,
                    checkpoint=self.config.reranker.checkpoint,
                    revision=(
                        self.reranker.resolved_revision
                        if self.reranker is not None
                        else None
                    ),
                ),
                "conversation": ModelStatus(
                    enabled=self.enable_conversation,
                    loaded=conversation_loaded,
                    checkpoint=self.config.conversation.checkpoint,
                    revision=(
                        self.conversation.revision
                        if self.conversation is not None
                        else None
                    ),
                ),
            },
        )


def _env_bool(name: str) -> bool:
    value = os.getenv(name, "true").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


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
