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
from llm.config import LLMServiceConfig
from offline.enrichment.ocr.adapters.florence import FlorenceAdapter
from offline.enrichment.ocr.config import OCRConfig


# Return Any because pydantic validates these rows into ``_ReadinessModel``.

from offline.enrichment.ocr.models.entities import OCRResult
from offline.enrichment.object_detection import load_vocab, normalized_boxes
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
        text_encoder: Any | None = None,
        captioner: Any | None = None,
        ocr_adapter: Any | None = None,
        text_generator: Any | None = None,
        *,
        enable_caption: bool = True,
        enable_visual_embedding: bool = True,
        enable_text_embedding: bool = True,
        enable_ocr: bool = False,
        enable_objects: bool = False,
        enable_asr: bool = False,
        enable_diarization: bool = False,
        enable_text_generation: bool = False,
        transcript_config: TranscriptJobConfig | None = None,
    ) -> None:
        self.config = config
        self.transcript_config = transcript_config
        self.enable_caption = enable_caption
        self.enable_visual_embedding = enable_visual_embedding
        self.enable_text_embedding = enable_text_embedding
        self.enable_ocr = enable_ocr
        self.enable_objects = enable_objects
        self.enable_asr = enable_asr
        self.enable_diarization = enable_diarization
        self.enable_text_generation = enable_text_generation

        self.asr = None
        self.object_detector = None
        self.diarization = None

        self.visual_encoder = visual_encoder or (
            cast(Any, EmbeddingService.create_visual_adapter(config.visual_embedding))
            if enable_visual_embedding
            else None
        )
        self.text_encoder = text_encoder or (
            cast(Any, EmbeddingService.create_text_adapter(config.caption_embedding))
            if enable_text_embedding
            else None
        )
        self.captioner = captioner or (
            cast(
                Any, EnrichmentService.create_caption_adapter(cast(Any, config.caption_generation))
            )
            if enable_caption
            else None
        )
        self.ocr_adapter: Any = ocr_adapter or (
            FlorenceAdapter(
                OCRConfig(
                    device=config.caption_generation.device,
                    dtype=config.caption_generation.dtype,
            ))
            if enable_ocr
            else None
        )
        if text_generator is not None:
            self.text_generator = text_generator
        elif enable_text_generation:
            from llm.local.text_generation import TextGenerationAdapter

            self.text_generator = TextGenerationAdapter(config.text_generation)
        else:
            self.text_generator = None

    @classmethod
    def from_environment(cls) -> LocalAdapter:
        path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
        config = LLMServiceConfig.from_yaml(path)

        enrichment_path = Path(os.getenv("HCMAI_ENRICHMENT_CONFIG", "configs/prepare.yaml"))
        transcript_config = (
            TranscriptJobConfig.from_yaml(enrichment_path) if enrichment_path.exists() else None
        )

        flags = _profile_flags()
        return cls(
            config,
            enable_caption=_env_bool("HCMAI_ENABLE_CAPTION", flags["caption"]),
            enable_visual_embedding=_env_bool("HCMAI_ENABLE_VISUAL_EMBEDDING", flags["visual_embedding"]),
            enable_text_embedding=_env_bool("HCMAI_ENABLE_TEXT_EMBEDDING", flags["text_embedding"]),
            enable_ocr=_env_bool("HCMAI_ENABLE_OCR", flags["ocr"]),
            enable_objects=_env_bool("HCMAI_ENABLE_OBJECTS", flags["objects"]),
            enable_asr=_env_bool("HCMAI_ENABLE_ASR", flags["asr"]),
            enable_diarization=_env_bool("HCMAI_ENABLE_DIARIZATION", flags["diarization"]),
            enable_text_generation=_env_bool("HCMAI_ENABLE_TEXT_GENERATION", flags["text_generation"]),
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
        if self.text_encoder is not None:
            self.text_encoder._load_model()
        if self.ocr_adapter is not None:
            self.ocr_adapter._load()
        if self.text_generator is not None:
            self.text_generator.load()

        if self.enable_objects:
            from ultralytics import YOLOE

            object_config = self.config.object_detection
            self.object_detector = YOLOE(object_config.model)
            if object_config.vocab_path is not None:
                names = load_vocab(object_config.vocab_path)
                self.object_detector.set_classes(
                    names, self.object_detector.get_text_pe(names)
                )

        if self.enable_asr and self.transcript_config:
            from offline.enrichment.transcripts.adapters.asr import ASRAdapter

            self.asr = ASRAdapter(self.transcript_config.asr)
            self.asr._load_asr()

        if self.enable_diarization and self.transcript_config:
            from offline.enrichment.transcripts.adapters.diarization import DiarizationAdapter

            self.diarization = DiarizationAdapter(self.transcript_config.diarization)
            self.diarization._load_pipeline()

    def generate_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        response_schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate text conforming to response_schema using the process-local text generator."""
        if self.text_generator is None:
            raise RuntimeError("text generation model is disabled")
        return self.text_generator.generate(
            messages,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text with the hosted evidence embedding model."""
        if self.text_encoder is None:
            raise RuntimeError("text embedding model is disabled")
        return self.text_encoder.encode_text(list(texts))

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

    def objects(
        self,
        images: Sequence[Image.Image],
        *,
        min_confidence: float | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, list[Any]]]:
        """Run YOLOE and return the canonical raw parallel-array payload."""

        if self.object_detector is None:
            raise RuntimeError("object detection model is disabled")
        config = self.config.object_detection
        confidence = config.min_confidence if min_confidence is None else float(min_confidence)
        maximum = config.top_k if top_k is None else int(top_k)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if maximum < 1:
            raise ValueError("top_k must be positive")

        results = self.object_detector.predict(
            list(images),
            conf=confidence,
            device=config.device,
            verbose=False,
        )
        if len(results) != len(images):
            raise RuntimeError("YOLOE returned the wrong result count")

        payloads: list[dict[str, list[Any]]] = []
        for result in results:
            height, width = result.orig_shape
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                payloads.append({
                    "detection_class_entities": [],
                    "detection_scores": [],
                    "detection_boxes": [],
                })
                continue
            order = boxes.conf.argsort(descending=True)[:maximum]
            payloads.append({
                "detection_class_entities": [
                    str(result.names[int(index)])
                    for index in boxes.cls[order].tolist()
                ],
                "detection_scores": [
                    min(1.0, max(0.0, float(score)))
                    for score in boxes.conf[order].tolist()
                ],
                "detection_boxes": normalized_boxes(
                    boxes.xyxy[order].tolist(), width, height
                ),
            })
        return payloads

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

    def transcribe_file(self, audio_path: str | Path, video_id: str):
        """Transcribe one already-uploaded local audio file."""

        if self.asr is None:
            raise RuntimeError("ASR capability is disabled")
        return self.asr.transcribe(Path(audio_path), video_id)

    def diarize_reference(self, payload: Any):
        """Download audio and assign speakers to supplied transcript segments."""

        if self.diarization is None:
            raise RuntimeError("diarization capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            download_audio(payload, path)
            return self.diarization.assign_speakers(path, payload.segments)


def _profile_flags() -> dict[str, bool]:
    """Resolve safe defaults for the core or single-A6000 GPU deployment."""

    profile = os.getenv("HCMAI_LLM_PROFILE", "manual").strip().lower()
    base = {
        "caption": False,
        "visual_embedding": False,
        "text_embedding": False,
        "ocr": False,
        "objects": False,
        "asr": False,
        "diarization": False,
        "text_generation": False,
    }
    if profile == "manual":
        return base
    if profile == "core":
        base.update(visual_embedding=True, text_embedding=True)
        return base
    if profile == "gpu":
        task = os.getenv("HCMAI_GPU_TASK", "caption").strip().lower()
        aliases = {"caption": "caption", "ocr": "ocr", "objects": "objects", "asr": "asr"}
        try:
            base[aliases[task]] = True
        except KeyError as error:
            raise ValueError("HCMAI_GPU_TASK must be one of: caption, ocr, objects, asr") from error
        return base
    raise ValueError("HCMAI_LLM_PROFILE must be one of: manual, core, gpu")


def _env_bool(name: str, default: bool = False) -> bool:
    """Read an explicit capability flag, defaulting hosted models to off.

    A service must opt into every model it owns. This prevents a narrowly
    configured ASR or Caption/OCR process from loading unrelated embedding checkpoints
    during application startup.
    """
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"
