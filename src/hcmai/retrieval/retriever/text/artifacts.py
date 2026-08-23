"""Configured pipeline for building frame-text retrieval artifacts."""

from __future__ import annotations

from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.retrieval.embedding.pipeline import EmbeddingService, TextEmbeddingAdapter
from thundercompute.pipeline import LLMServiceConfig
from hcmai.retrieval.retriever.artifacts import fingerprint_files
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.text.retriever import (
    build_context_index,
    build_text_index,
)

logger = get_logger(__name__)
_TEXT_SOURCES = {
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
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
    encoder: TextEmbeddingAdapter | None,
    source: RetrievalSource = RetrievalSource.CAPTION,
) -> TextEmbeddingAdapter:
    """Resolve the configured local or remote encoder unless one was injected."""

    selected = encoder
    if selected is None and settings.inference.enabled:
        import os
        from thundercompute.pipeline import LLMService

        base_url = os.getenv(
            "HCMAI_INFERENCE_BASE_URL", settings.inference.base_url
        )
        service = LLMService.remote(base_url, settings.inference)
        selected = EmbeddingService.create_remote_adapter(
            service,
            models.caption_embedding,
            embedding_dim=1024,
            source=source.value,
        )
    if selected is None:
        selected = EmbeddingService.create_text_adapter(models.caption_embedding)
    if selected.config.model_name != models.caption_embedding.model_name:
        raise ValueError(
            "Text encoder does not match thundercompute/config.yaml: "
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


def _context_artifact_paths(
    settings: AppConfig,
    context_path: str | Path | None,
    frames_path: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve the independent FrameContext input and index destinations."""

    context = _input_file(
        context_path or settings.dataset.enrichment.context_path,
        "CONTEXT enrichment",
    )
    frames = _input_file(
        frames_path or settings.dataset.frames_path, "Canonical frame metadata"
    )
    output = Path(output_dir or settings.index.context_path)
    return context, frames, output


def build_text_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "thundercompute/config.yaml",
    *,
    source: RetrievalSource = RetrievalSource.CAPTION,
    enrichment_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEmbeddingAdapter | None = None,
) -> DenseIndex:
    """Load configured inputs and build one frame-text retrieval index."""

    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text source {source.value!r}")
    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    enrichment, frames, output = _artifact_paths(
        settings, source, enrichment_path, frames_path, output_dir
    )
    selected_encoder = _text_encoder(
        settings, models, encoder, source=source
    )
    data = DataService.load(frames, {source: enrichment})
    index = build_text_index(
        data,
        selected_encoder,
        source,
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


def build_context_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "thundercompute/config.yaml",
    *,
    context_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEmbeddingAdapter | None = None,
) -> DenseIndex:
    """Build the deterministic FrameContext BGE index as a frame-native bundle.

    Context is a separate corpus from legacy caption/OCR/frame-ASR indexes, so
    this boundary deliberately does not route through ``build_text_artifacts``.
    """

    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    context, frames, output = _context_artifact_paths(
        settings, context_path, frames_path, output_dir
    )
    manifest = _input_file(
        context.with_name("manifest.json"), "CONTEXT manifest"
    )
    selected_encoder = _context_encoder(settings, models, encoder)
    data = DataService.load(frames, context_path=context)
    index = build_context_index(
        data,
        selected_encoder,
        output,
        embeddings_filename=settings.index.context_embedding_filename,
        dataset_version=settings.dataset.version,
        index_type=settings.index.type,
        source_fingerprint=fingerprint_files([context, manifest]),
    )
    logger.info(
        "CONTEXT index ready output=%s vectors=%d model=%s dimension=%d",
        output,
        index.metadata.vector_count,
        index.metadata.model_name,
        index.metadata.embedding_dim,
    )
    return index


def _context_encoder(
    settings: AppConfig,
    models: LLMServiceConfig,
    encoder: TextEmbeddingAdapter | None,
) -> TextEmbeddingAdapter:
    """Resolve the evidence encoder and preserve the hosted text family."""

    encoder_config = models.resolved_evidence_embedding
    selected = encoder
    if selected is None and settings.inference.enabled:
        import os
        from thundercompute.pipeline import LLMService

        base_url = os.getenv(
            "HCMAI_INFERENCE_BASE_URL", settings.inference.base_url
        )
        service = LLMService.remote(base_url, settings.inference)
        # The hosted contract selects its BGE-compatible endpoint by family;
        # "context" is an evidence label, not a remote embedding family.
        selected = EmbeddingService.create_remote_adapter(
            service,
            encoder_config,
            embedding_dim=1024,
            source="text",
        )
    if selected is None:
        selected = EmbeddingService.create_text_adapter(encoder_config)
    if selected.config.model_name != encoder_config.model_name:
        raise ValueError(
            "Context encoder does not match resolved evidence embedding: "
            f"{selected.config.model_name!r} != {encoder_config.model_name!r}"
        )
    return selected
