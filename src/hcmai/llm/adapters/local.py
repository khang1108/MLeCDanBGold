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
from hcmai.data.enrichment.ocr.models.entities import OCRResult
from hcmai.data.enrichment.pipeline import EnrichmentService
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.common.config import TranscriptJobConfig
from hcmai.llm.adapters.vqa import GroundedVQAModel
from hcmai.llm.pipeline import LLMServiceConfig
from hcmai.retrieval.reranking.pipeline import QwenRerankerConfig, RerankingService


class LocalAdapter:
    """Tải và quản lý vòng đời của các mô hình Machine Learning chạy trên máy cục bộ (local).
    Cung cấp các API chuẩn hóa để Inference Service có thể gọi (Embedding, VQA, OCR, Captioning).
    """

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
        enable_asr: bool = False,
        enable_diarization: bool = False,
        enable_transnet: bool = False,
        enable_gebd: bool = False,
        enable_dino: bool = False,
        preparation_config: S3CorpusPreparationConfig | None = None,
        transcript_config: TranscriptJobConfig | None = None,
    ) -> None:
        self.config = config
        self.preparation_config = preparation_config
        self.transcript_config = transcript_config
        self.enable_caption = enable_caption
        self.enable_visual_embedding = enable_visual_embedding
        self.enable_caption_embedding = enable_caption_embedding
        self.enable_reranker = enable_reranker
        self.enable_vqa = enable_vqa
        self.enable_ocr = enable_ocr
        self.enable_asr = enable_asr
        self.enable_diarization = enable_diarization
        self.enable_transnet = enable_transnet
        self.enable_gebd = enable_gebd
        self.enable_dino = enable_dino

        self.transnet = None
        self.gebd = None
        self.dino = None
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
        
        prep_path = Path(os.getenv("HCMAI_PREPARATION_CONFIG", "configs/preparation.s3.yaml"))
        preparation_config = S3CorpusPreparationConfig.from_yaml(prep_path) if prep_path.exists() else None
        
        enrichment_path = Path(os.getenv("HCMAI_ENRICHMENT_CONFIG", "configs/enrichment.yaml"))
        transcript_config = TranscriptJobConfig.from_yaml(enrichment_path) if enrichment_path.exists() else None

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
            enable_asr=_env_bool("HCMAI_ENABLE_ASR", default=False),
            enable_diarization=_env_bool("HCMAI_ENABLE_DIARIZATION", default=False),
            enable_transnet=_env_bool("HCMAI_ENABLE_TRANSNET", default=False),
            enable_gebd=_env_bool("HCMAI_ENABLE_GEBD", default=False),
            enable_dino=_env_bool("HCMAI_ENABLE_DINO", default=False),
            preparation_config=preparation_config,
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

        if self.enable_transnet and self.preparation_config:
            from hcmai.data.preprocessing.models import TransNetDetector
            self.transnet = TransNetDetector(self.preparation_config.preprocessing)
            self.transnet._load()

        if self.enable_gebd and self.preparation_config:
            from hcmai.data.preprocessing.models import EfficientGEBDDetector
            self.gebd = EfficientGEBDDetector(self.preparation_config.preprocessing)
            self.gebd._load()

        if self.enable_dino and self.preparation_config:
            from hcmai.data.preprocessing.selection import DinoEncoder
            self.dino = DinoEncoder(self.preparation_config.preprocessing)
            probe = Image.new("RGB", (self.preparation_config.preprocessing.efficientgebd_resolution,) * 2)
            try:
                self.dino.encode([probe])
            finally:
                probe.close()

        if self.enable_asr and self.transcript_config:
            from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter
            self.asr = ASRAdapter(self.transcript_config.asr)
            self.asr._load_asr()

        if self.enable_diarization and self.transcript_config:
            from hcmai.data.enrichment.transcripts.adapters.diarization import DiarizationAdapter
            self.diarization = DiarizationAdapter(self.transcript_config.diarization)
            self.diarization._load_pipeline()


    def embed_text(self, texts: list[str], source: str = "visual") -> np.ndarray:
        encoder = (
            self.caption_encoder if source == "text" else self.visual_encoder
        )
        if encoder is None:
            raise RuntimeError("embedding model is disabled")
        return encoder.encode_text(texts)

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
        transnet_loaded = self.transnet is not None
        gebd_loaded = self.gebd is not None
        dino_loaded = self.dino is not None
        asr_loaded = self.asr is not None
        diarization_loaded = self.diarization is not None
        
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
                    revision=self.config.visual_embedding.revision,
                ),
                "caption_embedding": ModelStatus(
                    enabled=self.enable_caption_embedding,
                    loaded=caption_loaded,
                    checkpoint=self.config.caption_embedding.model_name,
                    revision=self.config.caption_embedding.revision,
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
                    revision=(
                        self.ocr_adapter.config.revision
                        if self.ocr_adapter is not None
                        else None
                    ),
                ),
                "transnet": ModelStatus(
                    enabled=self.enable_transnet,
                    loaded=transnet_loaded,
                    checkpoint=self.preparation_config.remote_inference.transnet_model.model_name if self.preparation_config and self.preparation_config.remote_inference.transnet_model else None,
                    revision=self.preparation_config.remote_inference.transnet_model.revision if self.preparation_config and self.preparation_config.remote_inference.transnet_model else None,
                ),
                "efficientgebd": ModelStatus(
                    enabled=self.enable_gebd,
                    loaded=gebd_loaded,
                    checkpoint=self.preparation_config.remote_inference.efficientgebd_model.model_name if self.preparation_config and self.preparation_config.remote_inference.efficientgebd_model else None,
                    revision=self.preparation_config.remote_inference.efficientgebd_model.revision if self.preparation_config and self.preparation_config.remote_inference.efficientgebd_model else None,
                ),
                "dino": ModelStatus(
                    enabled=self.enable_dino,
                    loaded=dino_loaded,
                    checkpoint=self.preparation_config.preprocessing.dino_model if self.preparation_config else None,
                ),
                "asr": ModelStatus(
                    enabled=self.enable_asr,
                    loaded=asr_loaded,
                    checkpoint=self.transcript_config.asr.model_name if self.transcript_config else None,
                    revision=self.transcript_config.asr.revision if self.transcript_config else None,
                ),
                "diarization": ModelStatus(
                    enabled=self.enable_diarization,
                    loaded=diarization_loaded,
                    checkpoint=self.transcript_config.diarization.model_name if self.transcript_config else None,
                    revision=self.transcript_config.diarization.revision if self.transcript_config else None,
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
                shot_detection=transnet_loaded,
                event_detection=gebd_loaded,
                dino_embedding=dino_loaded,
                image_embedding=visual_loaded,
                caption=generator_loaded,
                ocr=ocr_loaded,
                asr=asr_loaded,
                diarization=diarization_loaded,
            ),
        )

    def boundary_scores(self, frames: np.ndarray, *, source: str) -> np.ndarray:
        if source == "shot":
            if self.transnet is None:
                raise RuntimeError("TransNet capability is disabled")
            return self.transnet.score(Path("remote-tensor"), frames)
        if self.gebd is None:
            raise RuntimeError("GEBD capability is disabled")
        import torch
        from hcmai.data.preprocessing.models import IMAGE_MEAN, IMAGE_STD

        tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255
        mean = torch.tensor(IMAGE_MEAN)[None, :, None, None]
        std = torch.tensor(IMAGE_STD)[None, :, None, None]
        return self.gebd._predict(list((tensor - mean) / std), len(frames))

    def transcribe_reference(self, payload: Any):
        import tempfile
        if self.asr is None:
            raise RuntimeError("ASR capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            _download_audio(payload, path)
            return self.asr.transcribe(path, payload.video_id)

    def diarize_reference(self, payload: Any):
        import tempfile
        if self.diarization is None:
            raise RuntimeError("diarization capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            _download_audio(payload, path)
            return self.diarization.assign_speakers(path, payload.segments)


def _download_audio(payload: Any, target: Path) -> None:
    import hashlib
    import httpx
    maximum = int(os.getenv("HCMAI_MAX_AUDIO_BYTES", str(1024 * 1024 * 1024)))
    digest = hashlib.sha256()

    total = 0
    with httpx.stream("GET", payload.audio_url, timeout=300, follow_redirects=False) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise ValueError("remote audio exceeds configured byte limit")
                digest.update(chunk)
                handle.write(chunk)
    
    if digest.hexdigest() != payload.audio_sha256:
        raise ValueError("remote audio checksum mismatch")


def _env_bool(name: str, default: bool = True) -> bool:
    """Hàm hỗ trợ đọc biến môi trường dạng boolean (true/false)."""
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
