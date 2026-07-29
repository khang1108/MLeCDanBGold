"""Configured pipeline for building caption retrieval artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.utils.logging import get_logger
from hcmai.data import CaptionStore, FrameStore
from hcmai.llm.config import LLMServiceConfig
from hcmai.retriever.caption.retriever import build_caption_index
from hcmai.retriever.dense import DenseEncoder, DenseIndex, TextEncoder

logger = get_logger(__name__)


def _input_file(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} path is not configured")
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} artifact is not available at {path}")
    return path


def build_caption_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "llm/config.yaml",
    *,
    captions_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEncoder | None = None,
) -> DenseIndex:
    """Load configured inputs and build all caption retrieval artifacts."""

    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    captions = _input_file(
        captions_path or settings.dataset.enrichment.caption_path,
        "Caption enrichment",
    )
    frames = _input_file(
        frames_path or settings.dataset.frames_path,
        "Canonical frame metadata",
    )
    output = Path(output_dir or settings.index.caption_path)
    selected_encoder = encoder
    if selected_encoder is None:
        local_config = models.caption_embedding.model_copy(
            update={
                "device": settings.inference.local_fallback_device,
                "batch_size": settings.inference.local_fallback_batch_size,
            }
        )
        local = DenseEncoder(local_config)
        if settings.inference.enabled:
            from hcmai.llm.client import InferenceClient, RemoteDenseEncoder

            client = InferenceClient(
                os.getenv(
                    "HCMAI_INFERENCE_BASE_URL",
                    settings.inference.base_url,
                ),
                settings.inference.timeout_seconds,
            )
            fallback = (
                local if settings.inference.local_embedding_fallback else None
            )
            selected_encoder = RemoteDenseEncoder(
                client,
                models.caption_embedding,
                embedding_dim=0,
                fallback=fallback,
                source="caption",
            )
        else:
            selected_encoder = local
    if selected_encoder.config.model_name != models.caption_embedding.model_name:
        raise ValueError(
            "Caption encoder does not match llm/config.yaml: "
            f"{selected_encoder.config.model_name!r} != "
            f"{models.caption_embedding.model_name!r}"
        )
    index = build_caption_index(
        CaptionStore(captions),
        FrameStore(frames),
        selected_encoder,
        output,
        dataset_version=settings.dataset.version,
        index_type=settings.index.type,
    )
    logger.info(
        "Caption index ready output=%s vectors=%d model=%s dimension=%d",
        output,
        index.metadata.vector_count,
        index.metadata.model_name,
        index.metadata.embedding_dim,
    )
    return index
