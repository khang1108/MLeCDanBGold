"""Kaggle T4 capability runtime for the HCMAI offline inference API."""

from __future__ import annotations

import hashlib
import os
import tempfile

from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from PIL import Image

from hcmai.common.schemas import (
    AudioReferenceRequest,
    DiarizationRequest,
    InferenceCapabilities,
    InferenceReadiness,
    ModelStatus,
)
from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.common.config import TranscriptJobConfig
from hcmai.llm.adapters.local import LocalAdapter
from hcmai.llm.config import LLMServiceConfig
from hcmai.llm.server.api import create_llm_app

_MODEL_NAMES = {
    "transnet",
    "gebd",
    "dino",
    "caption",
    "ocr",
    "visual_emb",
    "text_emb",
    "asr",
    "diarization",
}

def _models(value: str | None = None) -> frozenset[str]:
    """Get models name"""
    enabled = frozenset(
        item.strip().lower()
        for item in (value if value is not None else os.getenv("MODELS", "")).split(",")
        if item.strip()
    )
    unknown = enabled - _MODEL_NAMES

    if unknown:
        raise ValueError("unsupported MODELS capabilities: " + ", ".join(sorted(unknown)))
    if not enabled:
        raise ValueError("MODELS must enable at least one capability")
    return enabled


def _download_audio(payload: AudioReferenceRequest, target: Path) -> None:
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


class KaggleRuntime:
    """Compose only the pinned model capabilities assigned to one notebook."""

    def __init__(
        self,
        preparation: S3CorpusPreparationConfig,
        models: LLMServiceConfig,
        transcript_job: TranscriptJobConfig,
        enabled: frozenset[str],
        *,
        audio_fetcher: Callable[[AudioReferenceRequest, Path], None] = _download_audio,
    ) -> None:
        self.preparation = preparation
        self.config = models
        self.transcript_job = transcript_job
        self.enabled = enabled
        self.audio_fetcher = audio_fetcher
        self._validate_pins()
        self.base = LocalAdapter(
            models,
            enable_caption="caption" in enabled,
            enable_visual_embedding="visual_emb" in enabled,
            enable_caption_embedding="text_emb" in enabled,
            enable_reranker=False,
            enable_vqa=False,
            enable_ocr="ocr" in enabled,
        )
        self.captioner = self.base.captioner
        self.reranker = None
        self.vqa_model = None
        self.transnet = self.gebd = self.dino = None
        self.asr = self.diarization = None

    def _validate_pins(self) -> None:
        """Validate the pins."""
        pins = self.preparation.models
        checks = {
            "caption": (
                self.config.caption_generation.model_checkpoint,
                self.config.caption_generation.revision,
                pins.caption.model_name,
                pins.caption.revision,
            ),
            # caption and OCR share the same Florence model via caption_generation
            "ocr": (
                self.config.caption_generation.model_checkpoint,
                self.config.caption_generation.revision,
                pins.ocr.model_name,
                pins.ocr.revision,
            ),
            "visual_emb": (
                self.config.visual_embedding.model_name,
                self.config.visual_embedding.revision,
                pins.visual_embedding.model_name,
                pins.visual_embedding.revision,
            ),
            "text_emb": (
                self.config.caption_embedding.model_name,
                self.config.caption_embedding.revision,
                pins.text_embedding.model_name,
                pins.text_embedding.revision,
            ),
            "asr": (
                self.transcript_job.asr.model_name,
                self.transcript_job.asr.revision,
                pins.asr.model_name,
                pins.asr.revision,
            ),
            "diarization": (
                self.transcript_job.diarization.model_name,
                self.transcript_job.diarization.revision,
                pins.diarization.model_name,
                pins.diarization.revision,
            ),
        }

        mismatched = [
            capability
            for capability, (actual, revision, expected, expected_revision)
            in checks.items()
            if capability in self.enabled
            and (actual, revision) != (expected, expected_revision)
        ]

        if mismatched:
            raise ValueError(
                "Kaggle worker model pins differ from preparation config: "
                + ", ".join(sorted(mismatched))
            )

        if "transnet" in self.enabled and self.preparation.remote_inference.transnet_model is None:
            raise ValueError("TransNet worker requires remote_inference.transnet_model pin")

        if "gebd" in self.enabled and self.preparation.remote_inference.efficientgebd_model is None:
            raise ValueError("GEBD worker requires remote_inference.efficientgebd_model pin")

    @classmethod
    def from_environment(cls) -> KaggleRuntime:
        preparation_path = Path(
            os.getenv("HCMAI_PREPARATION_CONFIG", "configs/preparation.s3.yaml")
        )
        model_path = Path(os.getenv("HCMAI_MODEL_CONFIG", "llm/config.yaml"))
        enrichment_path = Path(
            os.getenv("HCMAI_ENRICHMENT_CONFIG", "configs/enrichment.yaml")
        )
        return cls(
            S3CorpusPreparationConfig.from_yaml(preparation_path),
            LLMServiceConfig.from_yaml(model_path),
            TranscriptJobConfig.from_yaml(enrichment_path),
            _models(),
        )

    def load(self) -> None:
        self.base.load()
        preprocessing = self.preparation.preprocessing

        if "transnet" in self.enabled:
            from hcmai.data.preprocessing.models import TransNetDetector

            self.transnet = TransNetDetector(preprocessing)
            self.transnet._load()

        if "gebd" in self.enabled:
            from hcmai.data.preprocessing.models import EfficientGEBDDetector

            self.gebd = EfficientGEBDDetector(preprocessing)
            self.gebd._load()

        if "dino" in self.enabled:
            from hcmai.data.preprocessing.selection import DinoEncoder

            self.dino = DinoEncoder(preprocessing)
            probe = Image.new("RGB", (preprocessing.efficientgebd_resolution,) * 2)
            try:
                self.dino.encode([probe])
            finally:
                probe.close()

        if "asr" in self.enabled:
            from hcmai.data.enrichment.transcripts.adapters.asr import ASRAdapter

            self.asr = ASRAdapter(self.transcript_job.asr)
            self.asr._load_asr()

        if "diarization" in self.enabled:
            from hcmai.data.enrichment.transcripts.adapters.diarization import (
                DiarizationAdapter,
            )

            self.diarization = DiarizationAdapter(self.transcript_job.diarization)
            self.diarization._load_pipeline()

    def readiness(self) -> InferenceReadiness:
        """Get inference readiness."""
        base = self.base.readiness()
        pins = self.preparation.models

        transnet_pin = self.preparation.remote_inference.transnet_model
        gebd_pin = self.preparation.remote_inference.efficientgebd_model

        models = dict(base.models)
        models.update({
            "transnet": ModelStatus(
                enabled="transnet" in self.enabled,
                loaded=self.transnet is not None,
                checkpoint=transnet_pin.model_name if transnet_pin else None,
                revision=transnet_pin.revision if transnet_pin else None,
            ),
            "efficientgebd": ModelStatus(
                enabled="gebd" in self.enabled,
                loaded=self.gebd is not None,
                checkpoint=gebd_pin.model_name if gebd_pin else None,
                revision=gebd_pin.revision if gebd_pin else None,
            ),
            "dino": ModelStatus(
                enabled="dino" in self.enabled,
                loaded=self.dino is not None,
                checkpoint=pins.dino.model_name,
                revision=pins.dino.revision,
            ),
            "asr": ModelStatus(
                enabled="asr" in self.enabled,
                loaded=self.asr is not None,
                checkpoint=pins.asr.model_name,
                revision=pins.asr.revision,
            ),
            "diarization": ModelStatus(
                enabled="diarization" in self.enabled,
                loaded=self.diarization is not None,
                checkpoint=pins.diarization.model_name,
                revision=pins.diarization.revision,
            ),
        })

        required = [status.loaded for status in models.values() if status.enabled]
        
        return InferenceReadiness(
            ready=bool(required) and all(required),
            models=models,
            capabilities=InferenceCapabilities(
                embedding=base.capabilities.embedding,
                shot_detection=self.transnet is not None,
                event_detection=self.gebd is not None,
                dino_embedding=self.dino is not None,
                image_embedding="visual_emb" in self.enabled,
                caption="caption" in self.enabled,
                ocr="ocr" in self.enabled,
                asr=self.asr is not None,
                diarization=self.diarization is not None,
            ),
        )

    def embed_text(self, texts: list[str], source: str = "visual") -> np.ndarray:
        return self.base.embed_text(texts, source)

    def embed_images(self, images: list[Image.Image], *, source: str) -> np.ndarray:
        if source == "dino":
            if self.dino is None:
                raise RuntimeError("DINO capability is disabled")
            return self.dino.encode(images)
        if self.base.visual_encoder is None:
            raise RuntimeError("visual embedding capability is disabled")
        return self.base.visual_encoder.encode_images(images)

    def caption(self, images: list[Image.Image]) -> list[str]:
        return self.base.caption(images)

    def ocr(self, images: list[Image.Image]) -> list[str]:
        return self.base.ocr(images)

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

    def transcribe_reference(self, payload: AudioReferenceRequest):
        if self.asr is None:
            raise RuntimeError("ASR capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            self.audio_fetcher(payload, path)
            return self.asr.transcribe(path, payload.video_id)

    def diarize_reference(self, payload: DiarizationRequest):
        if self.diarization is None:
            raise RuntimeError("diarization capability is disabled")
        with tempfile.TemporaryDirectory(prefix="hcmai-audio-") as directory:
            path = Path(directory) / f"{payload.video_id}.flac"
            self.audio_fetcher(payload, path)
            return self.diarization.assign_speakers(path, payload.segments)


def create_kaggle_app():
    """Uvicorn factory; model construction/loading happens in app lifespan."""

    return create_llm_app(KaggleRuntime.from_environment())
