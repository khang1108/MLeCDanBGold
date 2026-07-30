"""Configured pipeline for building frame-text retrieval artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.data import ASRStore, CaptionStore, FrameStore, OCRStore
from hcmai.llm.config import LLMServiceConfig
from hcmai.retriever.caption.retriever import build_text_index
from hcmai.retriever.dense import DenseIndex, TextEncoder, create_text_encoder

logger = get_logger(__name__)
_STORE_TYPES = {
    RetrievalSource.CAPTION: CaptionStore,
    RetrievalSource.OCR: OCRStore,
    RetrievalSource.ASR: ASRStore,
}


def _input_file(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} path is not configured")
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} artifact is not available at {path}")
    return path


def _text_encoder(
    settings: AppConfig,
    models: LLMServiceConfig,
    encoder: TextEncoder | None,
) -> TextEncoder:
    """Resolve the configured local or hosted BGE-M3 encoder."""

    selected = encoder
    if selected is None and settings.inference.enabled:
        from hcmai.llm.client import InferenceClient, RemoteDenseEncoder

        client = InferenceClient(
            os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
            settings.inference.timeout_seconds,
        )
        selected = RemoteDenseEncoder(
            client, models.caption_embedding, embedding_dim=0, source="caption"
        )
    if selected is None:
        selected = create_text_encoder(models.caption_embedding)
    if selected.config.model_name != models.caption_embedding.model_name:
        raise ValueError(
            "Text encoder does not match llm/config.yaml: "
            f"{selected.config.model_name!r} != "
            f"{models.caption_embedding.model_name!r}"
        )
    return selected


def _artifact_paths(
    settings: AppConfig,
    source: RetrievalSource,
    enrichment_path: str | Path | None,
    frames_path: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    configured = getattr(settings.dataset.enrichment, f"{source.value}_path")
    enrichment = _input_file(
        enrichment_path or configured, f"{source.value.upper()} enrichment"
    )
    frames = _input_file(
        frames_path or settings.dataset.frames_path, "Canonical frame metadata"
    )
    output = Path(output_dir or getattr(settings.index, f"{source.value}_path"))
    return enrichment, frames, output


def build_text_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "llm/config.yaml",
    *,
    source: RetrievalSource = RetrievalSource.CAPTION,
    enrichment_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEncoder | None = None,
) -> DenseIndex:
    """Load configured inputs and build one frame-text retrieval index."""

    if source not in _STORE_TYPES:
        raise ValueError(f"Unsupported text source {source.value!r}")
    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    enrichment, frames, output = _artifact_paths(
        settings, source, enrichment_path, frames_path, output_dir
    )
    selected_encoder = _text_encoder(settings, models, encoder)
    index = build_text_index(
        _STORE_TYPES[source](enrichment),
        FrameStore(frames),
        selected_encoder,
        output,
        embeddings_filename=settings.index.text_embedding_filenames[source],
        dataset_version=settings.dataset.version,
        index_type=settings.index.type,
    )
    logger.info(
        "%s index ready output=%s vectors=%d model=%s dimension=%d",
        source.value.upper(),
        output,
        index.metadata.vector_count,
        index.metadata.model_name,
        index.metadata.embedding_dim,
    )
    return index


def build_caption_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "llm/config.yaml",
    *,
    captions_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEncoder | None = None,
) -> DenseIndex:
    """Backward-compatible entry point for the caption text source."""

    return build_text_artifacts(
        config_path,
        model_config_path,
        source=RetrievalSource.CAPTION,
        enrichment_path=captions_path,
        frames_path=frames_path,
        output_dir=output_dir,
        encoder=encoder,
    )
