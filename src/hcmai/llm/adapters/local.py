"""Local model adapter used by the private inference service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from PIL import Image

from hcmai.common.schemas import (
    InferenceReadiness,
    ModelStatus,
    QuerySuggestion,
    VQAInferenceEvidence,
)
from hcmai.embedding.pipeline import EmbeddingService
from hcmai.enrichment.pipeline import EnrichmentService
from hcmai.llm.adapters.conversation import StructuredConversationModel
from hcmai.llm.pipeline import HostedConversationConfig, LLMServiceConfig
from hcmai.query_suggestions.pipeline import (
    parse_suggestions,
    suggestion_messages,
)
from hcmai.reranking.pipeline import QwenRerankerConfig, RerankingService


class LocalAdapter:
    """Load each configured model once and expose bounded inference methods."""

    def __init__(
        self,
        config: LLMServiceConfig,
        visual_encoder: Any | None = None,
        caption_encoder: Any | None = None,
        captioner: Any | None = None,
        reranker: Any | None = None,
        conversation: Any | None = None,
        query_suggester: Any | None = None,
        *,
        enable_caption: bool = True,
        enable_visual_embedding: bool = True,
        enable_caption_embedding: bool = True,
        enable_reranker: bool = True,
        enable_conversation: bool = True,
        enable_query_suggestions: bool | None = None,
    ) -> None:
        self.config = config
        self.enable_caption = enable_caption
        self.enable_visual_embedding = enable_visual_embedding
        self.enable_caption_embedding = enable_caption_embedding
        self.enable_reranker = enable_reranker
        self.enable_conversation = enable_conversation
        configured_suggestions = (
            config.query_suggestions.enabled
            and config.query_suggestions.active_provider == "gpu_inference"
        )
        self.enable_query_suggestions = (
            configured_suggestions
            if enable_query_suggestions is None
            else enable_query_suggestions and configured_suggestions
        )
        self.visual_encoder = visual_encoder or (
            cast(Any, EmbeddingService.create_text_adapter(config.visual_embedding))
            if enable_visual_embedding
            else None
        )
        self.caption_encoder = caption_encoder or (
            cast(Any, EmbeddingService.create_text_adapter(config.caption_embedding))
            if enable_caption_embedding
            else None
        )
        self.captioner = captioner or (
            cast(Any, EnrichmentService.create_caption_adapter(
                cast(Any, config.caption_generation)
            ))
            if enable_caption
            else None
        )
        self.reranker = reranker or (
            RerankingService.create_qwen_adapter(_reranker_config(config))
            if enable_reranker
            else None
        )
        self.conversation = conversation or (
            StructuredConversationModel(config.conversation)
            if enable_conversation
            else None
        )
        self.query_suggester = query_suggester or self._query_suggestion_model()

    @classmethod
    def from_environment(cls) -> LocalAdapter:
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
            enable_visual_embedding=_env_bool(
                "HCMAI_ENABLE_VISUAL_EMBEDDING"
            ),
            enable_caption_embedding=_env_bool(
                "HCMAI_ENABLE_CAPTION_EMBEDDING"
            ),
            enable_reranker=_env_bool("HCMAI_ENABLE_RERANKER"),
            enable_conversation=_env_bool("HCMAI_ENABLE_CONVERSATION"),
            enable_query_suggestions=_env_bool(
                "HCMAI_ENABLE_QUERY_SUGGESTIONS",
                default=config.query_suggestions.enabled,
            ),
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
        if (
            self.query_suggester is not None
            and self.query_suggester is not self.conversation
        ):
            self.query_suggester.load()

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
        return list(self.reranker.score_batch(query, images))

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.conversation is None:
            raise RuntimeError("conversation model is disabled")
        return self.conversation(request)

    def parse_trake(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.conversation is None:
            raise RuntimeError("conversation model is disabled")
        return self.conversation.structured_json(
            request["instruction"], request["raw_query"]
        )

    def suggest_queries(self, query: str, count: int) -> list[QuerySuggestion]:
        if self.query_suggester is None:
            raise RuntimeError("query-suggestion model is disabled")
        config = self.config.query_suggestions
        text = self.query_suggester.generate(
            suggestion_messages(query, count),
            max_new_tokens=config.generation.max_new_tokens,
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
        )
        return parse_suggestions(text, query, count)

    def answer_vqa(
        self, question: str, image: Image.Image, evidence: VQAInferenceEvidence
    ) -> str:
        if self.conversation is None:
            raise RuntimeError("vision-language model is disabled")
        return self.conversation.answer_vqa(question, image, evidence)

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
        suggestions_loaded = (
            self.query_suggester is not None
            and getattr(self.query_suggester, "model", None) is not None
        )
        return InferenceReadiness(
            ready=(not self.enable_caption or generator_loaded)
            and (not self.enable_visual_embedding or visual_loaded)
            and (not self.enable_caption_embedding or caption_loaded)
            and (not self.enable_reranker or reranker_loaded)
            and (not self.enable_conversation or conversation_loaded)
            and (not self.enable_query_suggestions or suggestions_loaded),
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
                    enabled=self.enable_visual_embedding,
                    loaded=visual_loaded,
                    checkpoint=self.config.visual_embedding.model_name,
                ),
                "caption_embedding": ModelStatus(
                    enabled=self.enable_caption_embedding,
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
                "query_suggestions": ModelStatus(
                    enabled=self.enable_query_suggestions,
                    loaded=suggestions_loaded,
                    checkpoint=(
                        self.config.query_suggestions.gpu_inference.checkpoint
                        if self.enable_query_suggestions
                        else None
                    ),
                    revision=(
                        self.query_suggester.revision
                        if self.query_suggester is not None
                        else None
                    ),
                ),
            },
        )

    def _query_suggestion_model(self) -> Any | None:
        if not self.enable_query_suggestions:
            return None
        values = self.config.query_suggestions.gpu_inference
        if self.conversation is not None and all(
            getattr(self.config.conversation, field) == getattr(values, field)
            for field in ("checkpoint", "revision", "device", "dtype")
        ):
            return self.conversation
        return StructuredConversationModel(HostedConversationConfig(
            checkpoint=values.checkpoint,
            revision=values.revision,
            device=values.device,
            dtype=values.dtype,
            max_new_tokens=(
                self.config.query_suggestions.generation.max_new_tokens
            ),
        ))


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
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
