"""Local model adapter used by the private inference service."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from PIL import Image
from hcmai.common.config import TranscriptJobConfig
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.retrieval.reranking.config import QwenRerankerConfig
from hcmai.retrieval.reranking.pipeline import RerankingService
from llm.config import LLMServiceConfig
from offline.enrichment.ocr.adapters.florence import FlorenceAdapter
from offline.enrichment.ocr.config import OCRConfig


# Return Any because pydantic validates these rows into ``_ReadinessModel``.

from offline.enrichment.ocr.models.entities import OCRResult
from offline.enrichment.pipeline import EnrichmentService
from llm.local.audio import download_audio
from llm.local.readiness import build_readiness


class LocalAdapter:
    """Tải và quản lý vòng đời của các mô hình Machine Learning chạy trên máy cục bộ (local).
    Cung cấp các API chuẩn hóa để Inference Service có thể gọi (Embedding, OCR, Captioning).
    """

    def __init__(
        self,
        config: LLMServiceConfig,
        visual_encoder: Any | None = None,
        caption_encoder: Any | None = None,
        captioner: Any | None = None,
        reranker: Any | None = None,
        query_preparer: Any | None = None,
        ocr_adapter: Any | None = None,
        *,
        enable_caption: bool = True,
        enable_visual_embedding: bool = True,
        enable_caption_embedding: bool = True,
        enable_reranker: bool = True,
        enable_query_preparation: bool = False,
        enable_ocr: bool = False,
        enable_asr: bool = False,
        enable_diarization: bool = False,
        transcript_config: TranscriptJobConfig | None = None,
    ) -> None:
        self.config = config
        self.transcript_config = transcript_config
        self.enable_caption = enable_caption
        self.enable_visual_embedding = enable_visual_embedding
        self.enable_caption_embedding = enable_caption_embedding
        self.enable_reranker = enable_reranker
        self.enable_query_preparation = enable_query_preparation
        self.enable_ocr = enable_ocr
        self.enable_asr = enable_asr
        self.enable_diarization = enable_diarization

        self.asr = None
        self.diarization = None

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
            cast(
                Any, EnrichmentService.create_caption_adapter(cast(Any, config.caption_generation))
            )
            if enable_caption
            else None
        )
        self.reranker = reranker or (
            RerankingService.create_qwen_adapter(_reranker_config(config))
            if enable_reranker
            else None
        )
        if query_preparer is not None:
            self.query_preparer = query_preparer
        elif enable_query_preparation:
            from llm.query_preparation import QwenQueryPreparer

            self.query_preparer = QwenQueryPreparer(config.query_preparation)
        else:
            self.query_preparer = None
        self.ocr_adapter: Any = ocr_adapter or (
            FlorenceAdapter(
                OCRConfig(
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

        enrichment_path = Path(os.getenv("HCMAI_ENRICHMENT_CONFIG", "configs/prepare.yaml"))
        transcript_config = (
            TranscriptJobConfig.from_yaml(enrichment_path) if enrichment_path.exists() else None
        )

        return cls(
            config,
            enable_caption=_env_bool("HCMAI_ENABLE_CAPTION"),
            enable_visual_embedding=_env_bool("HCMAI_ENABLE_VISUAL_EMBEDDING"),
            enable_caption_embedding=_env_bool("HCMAI_ENABLE_CAPTION_EMBEDDING"),
            enable_reranker=_env_bool("HCMAI_ENABLE_RERANKER"),
            enable_query_preparation=_env_bool("HCMAI_ENABLE_QUERY_PREPARATION", default=False),
            enable_ocr=_env_bool("HCMAI_ENABLE_OCR"),
            enable_asr=_env_bool("HCMAI_ENABLE_ASR", default=False),
            enable_diarization=_env_bool("HCMAI_ENABLE_DIARIZATION", default=False),
            transcript_config=transcript_config,
        )

    def load(self) -> None:
        """Tải các mô hình vào bộ nhớ (RAM/VRAM) trong suốt vòng đời của service.
        Việc tải mô hình chỉ thực hiện khi gọi hàm này, không thực hiện lúc import module
        để tiết kiệm tài nguyên và khởi động ứng dụng nhanh hơn.
        """
        if self.captioner is not None:
            self.captioner.resolve_revision()
        if self.visual_encoder is not None:
            self.visual_encoder._load_model()
        if self.caption_encoder is not None and self.caption_encoder is not self.visual_encoder:
            self.caption_encoder._load_model()
        if self.reranker is not None:
            self.reranker._ensure_loaded()
        if self.query_preparer is not None:
            self.query_preparer._ensure_loaded()
        if self.ocr_adapter is not None:
            self.ocr_adapter._load()

        if self.enable_asr and self.transcript_config:
            from offline.enrichment.transcripts.adapters.asr import ASRAdapter

            self.asr = ASRAdapter(self.transcript_config.asr)
            self.asr._load_asr()

        if self.enable_diarization and self.transcript_config:
            from offline.enrichment.transcripts.adapters.diarization import DiarizationAdapter

            self.diarization = DiarizationAdapter(self.transcript_config.diarization)
            self.diarization._load_pipeline()

    def embed_text(self, texts: list[str], source: str = "visual") -> np.ndarray:
        encoder = self.caption_encoder if source == "text" else self.visual_encoder
        if encoder is None:
            raise RuntimeError("embedding model is disabled")
        return encoder.encode_text(texts)

    def embed_images(
        self,
        images: Sequence[Image.Image],
        *,
        source: str = "visual",
        item_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Encode images into visual embedding vectors using the visual adapter."""
        if source != "visual":
            raise RuntimeError(f"image embedding does not support source {source!r}")
        if self.visual_encoder is None:
            raise RuntimeError("visual embedding model is disabled")
        if not hasattr(self.visual_encoder, "encode_images"):
            raise RuntimeError("visual encoder does not support image encoding")
        return self.visual_encoder.encode_images(list(images))

    def ocr(self, images: Sequence[Image.Image]) -> list[OCRResult]:
        """Return structured OCR results without discarding raw regions."""
        if self.ocr_adapter is None:
            raise RuntimeError("ocr model is disabled")
        return list(self.ocr_adapter.recognize_batch(images))

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

    def translate_query_events(self, events: list[str]) -> list[str]:
        """Translate query events with the process-owned Qwen model."""

        if self.query_preparer is None:
            raise RuntimeError("query-preparation model is disabled")
        return self.query_preparer.translate(events)

    def generate_query_candidates(
        self, events: list[str], candidate_count: int = 5
    ) -> dict[str, Any]:
        """Generate aligned candidates with the process-owned Qwen model."""

        if self.query_preparer is None:
            raise RuntimeError("query-preparation model is disabled")
        return self.query_preparer.generate_candidates(events, candidate_count)

    def readiness(self) -> Any:
        """Report enabled capability readiness and checkpoint provenance."""

        return build_readiness(self)

    @staticmethod
    def boundary_scores(frames: np.ndarray, *, source: str) -> np.ndarray:
        """Reject retired local boundary scoring explicitly."""
        raise RuntimeError(f"local {source} boundary scoring was removed; use BTC keyframes")

    def transcribe_reference(self, payload: Any):
        """Download and transcribe one checksum-validated audio reference."""

        if self.asr is None:
            raise RuntimeError("ASR capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            download_audio(payload, path)
            return self.asr.transcribe(path, payload.video_id)

    def diarize_reference(self, payload: Any):
        """Download audio and assign speakers to supplied transcript segments."""

        if self.diarization is None:
            raise RuntimeError("diarization capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            download_audio(payload, path)
            return self.diarization.assign_speakers(path, payload.segments)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an explicit capability flag, defaulting hosted models to off.

    A service must opt into every model it owns. This prevents a narrowly
    configured ASR or Caption/OCR process from loading unrelated embedding,
    reranking checkpoints during application startup.
    """
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
