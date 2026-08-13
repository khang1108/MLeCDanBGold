"""Local model adapter used by the private inference service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from PIL import Image

from hcmai.common.schemas import (
    InferenceCapabilities,
    InferenceReadiness,
    ModelStatus,
    VQAInferenceEvidence,
)
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.data.enrichment.ocr.adapters.florence import FlorenceAdapter
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.pipeline import EnrichmentService
from hcmai.llm.adapters.vqa import GroundedVQAModel
from hcmai.llm.pipeline import LLMServiceConfig
from hcmai.retrieval.reranking.pipeline import QwenRerankerConfig, RerankingService


class LocalAdapter:
    """Load each configured model once and expose bounded inference methods."""

    def __init__(
        self,
        config: LLMServiceConfig,
        visual_encoder: Any | None = None,
        caption_encoder: Any | None = None,
        captioner: Any | None = None,
        reranker: Any | None = None,
        vqa_model: Any | None = None,
        ocr_adapter: Any | None = None,
        *,
        enable_caption: bool = True,
        enable_visual_embedding: bool = True,
        enable_caption_embedding: bool = True,
        enable_reranker: bool = True,
        enable_vqa: bool = True,
        enable_ocr: bool = False,
    ) -> None:
        self.config = config
        self.enable_caption = enable_caption
        self.enable_visual_embedding = enable_visual_embedding
        self.enable_caption_embedding = enable_caption_embedding
        self.enable_reranker = enable_reranker
        self.enable_vqa = enable_vqa
        self.enable_ocr = enable_ocr
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
        self.vqa_model = vqa_model or (
            GroundedVQAModel(config.vqa_model)
            if enable_vqa
            else None
        )
        self.ocr_adapter: Any = ocr_adapter or (
            FlorenceAdapter(OCRConfig(
                checkpoint=config.caption_generation.model_checkpoint,
                revision=config.caption_generation.revision,
                device=config.caption_generation.device,
                dtype=config.caption_generation.dtype,
            ))
            if enable_ocr
            else None
        )

    @classmethod
    def from_environment(cls) -> LocalAdapter:
        path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
        config = LLMServiceConfig.from_yaml(path)
        checkpoint = os.getenv("HCMAI_VQA_MODEL")
        if checkpoint:
            vqa_model = config.vqa_model.model_copy(
                update={"checkpoint": checkpoint}
            )
            config = config.model_copy(update={"vqa_model": vqa_model})
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
            enable_vqa=_env_bool("HCMAI_ENABLE_VQA"),
            enable_ocr=_env_bool("HCMAI_ENABLE_OCR"),
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
        if self.vqa_model is not None:
            self.vqa_model.load()
        if self.ocr_adapter is not None:
            self.ocr_adapter._load()


    def embed_text(self, texts: list[str], source: str = "visual") -> np.ndarray:
        encoder = (
            self.caption_encoder if source == "text" else self.visual_encoder
        )
        if encoder is None:
            raise RuntimeError("embedding model is disabled")
        return encoder.encode_text(texts)

    def ocr(self, images: Sequence[Image.Image]) -> list[str]:
        if self.ocr_adapter is None:
            raise RuntimeError("ocr model is disabled")
        results = self.ocr_adapter.recognize_batch(images)
        return [str(r.text).strip() for r in results]

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

    def answer_vqa(
        self,
        question: str,
        image: Image.Image,
        evidence: VQAInferenceEvidence,
        scene_context: str = "",
    ) -> str:
        if self.vqa_model is None:
            raise RuntimeError("vision-language model is disabled")
        return self.vqa_model.answer_vqa(
            question, image, evidence, scene_context=scene_context
        )

    def answer_vqa_multi(
        self,
        question: str,
        images: list[Image.Image],
        frame_ids: list[str],
        evidence: VQAInferenceEvidence,
        scene_context: str = "",
    ) -> dict[str, Any]:
        if self.vqa_model is None:
            raise RuntimeError("vision-language model is disabled")
        return self.vqa_model.answer_vqa_multi(
            question, images, frame_ids, evidence, scene_context=scene_context
        )

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
        vqa_loaded = (
            self.vqa_model is not None and self.vqa_model.model is not None
        )
        ocr_loaded = (
            self.ocr_adapter is not None and self.ocr_adapter.model is not None
        )
        return InferenceReadiness(
            ready=(not self.enable_caption or generator_loaded)
            and (not self.enable_visual_embedding or visual_loaded)
            and (not self.enable_caption_embedding or caption_loaded)
            and (not self.enable_reranker or reranker_loaded)
            and (not self.enable_vqa or vqa_loaded)
            and (not self.enable_ocr or ocr_loaded),
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
                "vqa": ModelStatus(
                    enabled=self.enable_vqa,
                    loaded=vqa_loaded,
                    checkpoint=self.config.vqa_model.checkpoint,
                    revision=(
                        self.vqa_model.revision
                        if self.vqa_model is not None
                        else None
                    ),
                ),
                "ocr": ModelStatus(
                    enabled=self.enable_ocr,
                    loaded=ocr_loaded,
                    checkpoint=(
                        self.ocr_adapter.config.checkpoint
                        if self.ocr_adapter is not None
                        else None
                    ),
                ),
            },
            capabilities=InferenceCapabilities(
                embedding=visual_loaded or caption_loaded,
                reranking=reranker_loaded,
                multi_image_vqa=(
                    vqa_loaded
                    and bool(getattr(self.vqa_model, "supports_multi_image", False))
                ),
                structured_parsing=False,
            ),
        )


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
