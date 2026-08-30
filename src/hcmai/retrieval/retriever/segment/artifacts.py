"""Build and publish segment-native ASR retrieval artifacts offline.

This module embeds completed transcript text and preserves its timeline and ASR
provenance in a ``SegmentDenseIndex`` mapping. It intentionally does not load
canonical frames, materialize legacy frame-ASR evidence, or invent ``frame_id``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from hcmai.common.config import AppConfig
from hcmai.common.schemas import ProcessingStatus, RetrievalSource, TranscriptSegment
from hcmai.common.utils.logging import get_logger
from hcmai.data.enrichment.transcripts.artifacts import (
    load_transcript_artifact_records,
)
from thundercompute.config import LLMServiceConfig
from hcmai.retrieval.embedding.pipeline import EmbeddingService, TextEmbeddingAdapter
from hcmai.retrieval.retriever.artifacts import fingerprint_files, publish_directory
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from hcmai.retrieval.retriever.text.retriever import (
    _encode_texts,
    _encoder_revision,
    _normalized,
)

logger = get_logger(__name__)


def build_segment_corpus(
    records: tuple[TranscriptSegment, ...],
) -> tuple[list[str], pd.DataFrame]:
    """Serialize completed transcript speech into a segment-native text corpus.

    Whitespace is normalized without adding timestamps, speakers, or language
    to the embedded text. Timeline and provider provenance remain structured in
    the mapping, and non-completed segments are omitted rather than scored as
    negative evidence.
    """

    texts: list[str] = []
    rows: list[dict[str, object]] = []
    for segment in records:
        if segment.status is not ProcessingStatus.COMPLETED:
            continue
        text = " ".join(segment.text.split())
        if not text:
            continue
        rows.append(
            {
                "embedding_index": len(texts),
                "segment_id": segment.segment_id,
                "video_id": segment.video_id,
                "segment_index": segment.segment_index,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "language": segment.language,
                "speaker_id": segment.speaker_id,
                "confidence": segment.confidence,
                "status": segment.status.value,
                "model_name": segment.model_name,
                "model_revision": segment.model_revision,
                "artifact_version": segment.artifact_version,
            }
        )
        texts.append(text)
    if not texts:
        raise ValueError("Transcript artifact contains no usable completed segments")
    # Object dtype retains ``confidence=None`` as unknown evidence instead of
    # silently coercing it to a numeric zero.
    return texts, pd.DataFrame(rows, dtype=object)


def transcript_lineage_files(transcripts_path: str | Path) -> tuple[Path, ...]:
    """Return all transcript shards and present adjacent manifests in order.

    Each ``video.parquet`` is followed by ``video.manifest.json`` when that
    manifest exists. This captures the complete grouped transcript source while
    remaining compatible with legacy standalone Parquet fixtures.
    """

    path = Path(transcripts_path)
    parquet_files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    files: list[Path] = []
    for parquet in parquet_files:
        files.append(parquet)
        manifest = parquet.with_suffix(".manifest.json")
        if manifest.is_file():
            files.append(manifest)
    return tuple(files)


def build_asr_segment_index(
    records: tuple[TranscriptSegment, ...],
    encoder: TextEmbeddingAdapter,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
    source_fingerprint: str | None = None,
) -> SegmentDenseIndex:
    """Embed completed transcript text and publish one validated segment bundle."""

    artifact_name = _embedding_artifact_name(embeddings_filename)
    texts, mapping = build_segment_corpus(records)
    vectors = _normalized(_encode_texts(texts, encoder, RetrievalSource.ASR))
    if len(vectors) != len(mapping):
        raise ValueError(
            f"Text encoder returned {len(vectors)} vectors for {len(mapping)} segments"
        )

    index = SegmentDenseIndex.build(
        vectors,
        mapping,
        dataset_version=dataset_version,
        model_name=encoder.config.model_name,
        index_type=index_type,
    )
    index.metadata.model_revision = _encoder_revision(encoder)
    index.metadata.source_fingerprint = source_fingerprint
    _save_index_with_embeddings(index, output_dir, artifact_name, vectors)
    return index


def build_asr_segment_artifacts(
    config_path: str | Path = "configs/baseline.yaml",
    model_config_path: str | Path = "thundercompute/config.yaml",
    *,
    transcripts_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    encoder: TextEmbeddingAdapter | None = None,
) -> SegmentDenseIndex:
    """Build the configured segment-native ASR corpus and exact dense index.

    The input fingerprint covers every transcript Parquet shard and each
    present sibling ``.manifest.json`` file. Heavy embedding and publication
    happen only when this explicit offline boundary is called.
    """

    settings = AppConfig.from_yaml(config_path)
    models = LLMServiceConfig.from_yaml(model_config_path)
    configured_transcripts = transcripts_path or settings.dataset.enrichment.transcripts_path
    source_path, lineage_files = _transcript_source(configured_transcripts)
    output = Path(output_dir or settings.index.asr_segment_path)
    selected_encoder = _segment_encoder(settings, models, encoder)

    index = build_asr_segment_index(
        load_transcript_artifact_records(source_path),
        selected_encoder,
        output,
        embeddings_filename=settings.index.asr_segment_embedding_filename,
        dataset_version=settings.dataset.version,
        index_type=settings.index.type,
        source_fingerprint=fingerprint_files(lineage_files),
    )
    logger.info(
        "ASR segment index ready output=%s vectors=%d model=%s dimension=%d",
        output,
        index.metadata.vector_count,
        index.metadata.model_name,
        index.metadata.embedding_dim,
    )
    return index


def _transcript_source(
    value: str | Path | None,
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve a non-empty transcript Parquet source before loading models."""

    if value is None:
        raise ValueError("Transcript artifact path is not configured")
    path = Path(value)
    parquet_files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    if (
        not parquet_files
        or any(
            parquet.suffix != ".parquet"
            or not parquet.is_file()
            or parquet.stat().st_size == 0
            for parquet in parquet_files
        )
    ):
        raise FileNotFoundError(f"Transcript artifact is not available at {path}")
    return path, transcript_lineage_files(path)


def _segment_encoder(
    settings: AppConfig,
    models: LLMServiceConfig,
    encoder: TextEmbeddingAdapter | None,
) -> TextEmbeddingAdapter:
    """Resolve the evidence encoder while preserving hosted text-family routing."""

    encoder_config = models.resolved_evidence_embedding
    selected = encoder
    if selected is None and settings.inference.enabled:
        from thundercompute.pipeline import LLMService

        base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
        service = LLMService.remote(base_url, settings.inference)
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
            "ASR segment encoder does not match resolved evidence embedding: "
            f"{selected.config.model_name!r} != {encoder_config.model_name!r}"
        )
    return selected


def _embedding_artifact_name(embeddings_filename: str) -> Path:
    """Validate that supplemental vectors cannot escape their index bundle."""

    artifact_name = Path(embeddings_filename)
    if artifact_name.name != embeddings_filename or artifact_name.suffix != ".npy":
        raise ValueError("embeddings_filename must be a plain .npy filename")
    return artifact_name


def _save_index_with_embeddings(
    index: SegmentDenseIndex,
    output_dir: str | Path,
    embeddings_filename: Path,
    vectors: np.ndarray,
) -> Path:
    """Publish the validated segment index and supplemental vectors atomically."""

    output = Path(output_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        index.save(staged)
        np.save(staged / embeddings_filename, vectors)
        SegmentDenseIndex.load(staged)
        publish_directory(staged, output)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise
    return output


__all__ = [
    "build_asr_segment_artifacts",
    "build_asr_segment_index",
    "build_segment_corpus",
    "transcript_lineage_files",
]
