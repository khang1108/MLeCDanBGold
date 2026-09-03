"""Build and publish frame-native text and FrameContext index bundles offline.

This module joins offline evidence to canonical frame identity, embeds it, and
publishes the existing DenseIndex bundle layout.  Runtime retrieval only loads
and searches the resulting bundle through ``ContextRetriever``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Protocol

import numpy as np
import pandas as pd
from hcmai.common.config import AppConfig
from hcmai.common.utils.logging import get_logger
from hcmai.retrieval.embedding.models.contracts import TextEmbeddingAdapter
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.retriever.artifacts import fingerprint_files, publish_directory
from hcmai.retrieval.retriever.dense.index import DenseIndex
from llm.pipeline import LLMService, LLMServiceConfig
from offline.artifact_readers import (
    ASRArtifactReader,
    CaptionArtifactReader,
    FrameArtifactReader,
    FrameContextArtifactReader,
    OCRArtifactReader,
    assert_frame_identity,
    assert_matching_lineage,
)

from tqdm.auto import tqdm

logger = get_logger(__name__)
_TEXT_SOURCES = {
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
}


class _FrameIdentity(Protocol):
    """Canonical identity retained by frame and specialist artifact rows."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp_ms: int


class _FrameLookup(Protocol):
    """Canonical frame lookup needed by an offline frame-native index join."""

    @staticmethod
    def get(frame_id: str) -> _FrameIdentity:
        """Return one canonical frame for an exact internal identity."""


class _TextEvidenceLookup(Protocol):
    """Completed text and row iteration required by one specialist index join."""

    @staticmethod
    def get_text(frame_id: str) -> str | None:
        """Return usable text for one exact canonical frame identity."""

    @staticmethod
    def iter_records() -> Iterator[_FrameIdentity]:
        """Iterate every persisted specialist row in deterministic order."""


def build_text_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "llm/config.yaml",
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
    selected_encoder = _text_encoder(settings, models, encoder, source=source)
    frame_store = FrameArtifactReader(frames)
    evidence = _text_store(source, enrichment)
    index = build_text_index(
        frame_store,
        evidence,
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
    model_config_path: str | Path = "llm/config.yaml",
    *,
    context_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEmbeddingAdapter | None = None,
) -> DenseIndex:
    """Build the deterministic FrameContext BGE index as a frame-native bundle."""

    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    context, frames, output = _context_artifact_paths(
        settings, context_path, frames_path, output_dir
    )
    manifest = _input_file(context.with_name("manifest.json"), "CONTEXT manifest")
    selected_encoder = _context_encoder(settings, models, encoder)
    index = build_context_index(
        FrameArtifactReader(frames),
        FrameContextArtifactReader(context),
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


def build_text_index(
    frames: _FrameLookup,
    evidence: _TextEvidenceLookup,
    encoder: TextEmbeddingAdapter,
    source: RetrievalSource,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Build a text index from validated offline evidence and canonical frames."""

    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text evidence source {source.value!r}")
    artifact_name = _embedding_artifact_name(embeddings_filename)
    texts, mapping = _text_corpus(frames, evidence, source)
    vectors = _normalized(_encode_texts(texts, encoder, source))
    if len(vectors) != len(mapping):
        raise ValueError(f"Text encoder returned {len(vectors)} vectors for {len(mapping)} texts")
    index = DenseIndex.build(
        vectors,
        mapping,
        dataset_version=dataset_version,
        model_name=encoder.config.model_name,
        index_type=index_type,
        show_progress=True,
    )
    _save_index_with_embeddings(index, output_dir, artifact_name, vectors)
    return index


def build_context_index(
    frames: _FrameLookup,
    contexts: _TextEvidenceLookup,
    encoder: TextEmbeddingAdapter,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
    source_fingerprint: str | None = None,
) -> DenseIndex:
    """Build and safely publish the dedicated frame-native Context index."""

    artifact_name = _embedding_artifact_name(embeddings_filename)
    texts, mapping = _context_corpus(frames, contexts)
    vectors = _normalized(_encode_texts(texts, encoder, RetrievalSource.CONTEXT))
    if len(vectors) != len(mapping):
        raise ValueError(
            f"Text encoder returned {len(vectors)} vectors for {len(mapping)} contexts"
        )
    index = DenseIndex.build(
        vectors,
        mapping,
        dataset_version=dataset_version,
        model_name=encoder.config.model_name,
        index_type=index_type,
        show_progress=True,
    )
    index.metadata.retrieval_source = RetrievalSource.CONTEXT.value
    index.metadata.entity_kind = "frame"
    index.metadata.model_revision = _encoder_revision(encoder)
    index.metadata.source_fingerprint = source_fingerprint
    _save_index_with_embeddings(index, output_dir, artifact_name, vectors)
    return index


def _input_file(value: str | Path | None, label: str) -> Path:
    """Require a non-empty source artifact before any model is loaded."""

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

        base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
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
            "Text encoder does not match llm/config.yaml: "
            f"{selected.config.model_name!r} != {models.caption_embedding.model_name!r}"
        )
    return selected


def _artifact_paths(
    settings: AppConfig,
    source: RetrievalSource,
    enrichment_path: str | Path | None,
    frames_path: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve one source's configured input artifacts and output bundle."""

    configured = getattr(settings.dataset.enrichment, f"{source.value}_path")
    enrichment = _input_file(enrichment_path or configured, f"{source.value.upper()} enrichment")
    frames = _input_file(frames_path or settings.dataset.frames_path, "Canonical frame metadata")
    return enrichment, frames, Path(output_dir or getattr(settings.index, f"{source.value}_path"))


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
    frames = _input_file(frames_path or settings.dataset.frames_path, "Canonical frame metadata")
    return context, frames, Path(output_dir or settings.index.context_path)


def _text_store(
    source: RetrievalSource,
    artifact_path: Path,
) -> CaptionArtifactReader | OCRArtifactReader | ASRArtifactReader:
    """Open the offline specialist store required by one text index build."""

    stores = {
        RetrievalSource.CAPTION: CaptionArtifactReader,
        RetrievalSource.OCR: OCRArtifactReader,
        RetrievalSource.ASR: ASRArtifactReader,
    }
    try:
        return stores[source](artifact_path)
    except KeyError:
        raise ValueError(f"Unsupported text source {source.value!r}") from None


def _context_encoder(
    settings: AppConfig,
    models: LLMServiceConfig,
    encoder: TextEmbeddingAdapter | None,
) -> TextEmbeddingAdapter:
    """Resolve the evidence encoder and preserve hosted text-family routing."""

    encoder_config = models.resolved_evidence_embedding
    selected = encoder
    if selected is None and settings.inference.enabled:

        base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
        service = LLMService.remote(base_url, settings.inference)
        selected = EmbeddingService.create_remote_adapter(
            service, encoder_config, embedding_dim=1024, source="text"
        )
    if selected is None:
        selected = EmbeddingService.create_text_adapter(encoder_config)
    if selected.config.model_name != encoder_config.model_name:
        raise ValueError(
            "Context encoder does not match resolved evidence embedding: "
            f"{selected.config.model_name!r} != {encoder_config.model_name!r}"
        )
    return selected


def _text_corpus(
    frames: _FrameLookup,
    evidence: _TextEvidenceLookup,
    source: RetrievalSource,
) -> tuple[list[str], pd.DataFrame]:
    """Join usable enrichment text to canonical frame identity."""

    if source in {RetrievalSource.CAPTION, RetrievalSource.OCR}:
        assert_matching_lineage(frames, evidence, source.value.upper())

    texts: list[str] = []
    mapping: list[dict[str, Any]] = []
    for row in evidence.iter_records():
        frame_id = str(getattr(row, "frame_id"))
        frame = frames.get(frame_id)
        if source in {RetrievalSource.CAPTION, RetrievalSource.OCR}:
            # Validate every row before omitting no-text or failed evidence. A
            # malformed row must not be able to disappear from the index build.
            assert_frame_identity(row, frame, source.value.upper())
        text = evidence.get_text(frame_id)
        if text is None:
            continue
        mapping.append({
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "frame_idx": frame.frame_idx,
            "timestamp_ms": frame.timestamp_ms,
            "embedding_index": len(texts),
        })
        texts.append(text)
    if not texts:
        raise ValueError(f"{source.value} artifact contains no usable completed text")
    return texts, pd.DataFrame(mapping)


def _context_corpus(
    frames: _FrameLookup,
    contexts: _TextEvidenceLookup,
) -> tuple[list[str], pd.DataFrame]:
    """Join non-empty deterministic context text to canonical frame identity."""

    assert_matching_lineage(frames, contexts, "CONTEXT")
    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for context in contexts.iter_records():
        frame = frames.get(context.frame_id)
        # Context is a derived artifact but still carries the same canonical
        # frame identity. Validate it before excluding empty context text.
        assert_frame_identity(context, frame, "CONTEXT")
        text = contexts.get_text(context.frame_id)
        if text is None:
            continue
        rows.append({
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "frame_idx": frame.frame_idx,
            "timestamp_ms": frame.timestamp_ms,
            "embedding_index": len(texts),
        })
        texts.append(text)
    if not texts:
        raise ValueError("FrameContext artifact contains no usable context_text")
    return texts, pd.DataFrame(rows)


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Return finite float32 unit vectors suitable for inner product."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Text encoder must return a finite 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Text encoder returned a zero vector")
    return values / norms


def _encode_texts(
    texts: list[str],
    encoder: TextEmbeddingAdapter,
    source: RetrievalSource,
) -> np.ndarray:
    """Encode one text channel in the configured batch size with progress."""

    batch_size = encoder.config.batch_size
    batches: list[np.ndarray] = []
    with tqdm(
        total=len(texts),
        desc=f"Embedding {source.value}",
        unit="text",
        dynamic_ncols=True,
    ) as progress:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(encoder.encode_text(batch))
            progress.update(len(batch))
    return np.vstack(batches)


def _embedding_artifact_name(embeddings_filename: str) -> Path:
    """Reject supplemental vector paths that could escape their index bundle."""

    artifact_name = Path(embeddings_filename)
    if artifact_name.name != embeddings_filename or artifact_name.suffix != ".npy":
        raise ValueError("embeddings_filename must be a plain .npy filename")
    return artifact_name


def _save_index_with_embeddings(
    index: DenseIndex,
    output_dir: str | Path,
    embeddings_filename: Path,
    vectors: np.ndarray,
) -> Path:
    """Publish an index and supplemental vectors as one validated outer bundle."""

    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        index.save(staged)
        np.save(staged / embeddings_filename, vectors)
        DenseIndex.load(staged)
        publish_directory(staged, output)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise
    return output


def _encoder_revision(encoder: TextEmbeddingAdapter) -> str | None:
    """Extract optional pinned encoder revision for index provenance."""

    value = getattr(encoder, "resolved_revision", None)
    if value is None:
        value = getattr(encoder.config, "revision", None)
    return str(value) if value is not None else None


__all__ = ["build_context_artifacts", "build_text_artifacts"]
