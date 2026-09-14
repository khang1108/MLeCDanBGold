"""Load online retrieval indexes and temporal evidence at startup.

This module owns index loading, encoder selection, and the visual, Context,
ASR, and BM25 capabilities used by online search. It does not load canonical
frame metadata or construct HTTP-facing workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hcmai.common.config import AppConfig, resolve_repository_path
from hcmai.corpus import Corpus
from hcmai.inference import EmbeddingClient, load_embedding_endpoint
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.evidence.bm25 import BM25TemporalScorer
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
# pyrefly: ignore [missing-import]
from llm.config import LLMServiceConfig
# pyrefly: ignore [missing-import]
from llm.pipeline import LLMService


def load_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    llm: LLMService | None,
    messages: list[str],
    *,
    corpus: Corpus | None,
) -> RetrievalService | None:
    """Load the configured visual index and optional text retrieval indexes."""
    index_dir = _runtime_path("HCMAI_INDEX_PATH", settings.index.path)
    if not index_dir.is_dir():
        messages.append(f"Index directory not available at {index_dir}")
        return None
    try:
        visual = RetrievalService.load_index(index_dir)
        visual_encoder = _query_encoder(models.visual_embedding, visual, llm, "visual")
    except Exception as error:
        messages.append(
            f"Could not load required visual index {index_dir}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    return _load_fast_track_retrieval(
        settings,
        models,
        visual,
        visual_encoder,
        llm,
        corpus,
        messages,
    )


def select_visual_retriever(retrieval: RetrievalService | None) -> Any | None:
    """Select the configured visual retrieval capability for online services."""
    return (
        retrieval.source_retriever(RetrievalSource.VISUAL)
        if retrieval is not None
        else None
    )


def load_image_encoder(
    models: LLMServiceConfig,
    visual: Any | None,
    llm: LLMService | None,
    messages: list[str],
) -> Any | None:
    """Reuse a local image encoder or create its hosted visual adapter."""
    if visual is None:
        return None
    if hasattr(visual.encoder, "encode_images"):
        return visual.encoder
    if llm is None:
        messages.append("Image search unavailable: SigLIP2 image encoder missing")
        return None
    try:
        return EmbeddingService.create_remote_visual_adapter(
            llm,
            models.visual_embedding,
            visual.index.metadata.embedding_dim,
        )
    except Exception as error:
        messages.append(
            "Image search unavailable: " f"{type(error).__name__}: {error}"
        )
        return None


def load_temporal_evidence(
    settings: AppConfig,
    retrieval: RetrievalService | None,
    visual: Any | None,
    messages: list[str],
) -> TemporalEvidenceScorer | None:
    """Load independent full-corpus Dense and BM25 temporal capabilities."""
    if retrieval is None:
        return None
    if visual is None:
        messages.append("Temporal evidence unavailable: visual Dense index missing")
        return None

    dense, context_ready, asr_ready = _load_dense_temporal(
        settings,
        retrieval,
        visual,
        messages,
    )
    bm25 = _load_bm25_temporal(settings, visual, messages)
    if dense is None and bm25 is None:
        return None
    return TemporalEvidenceScorer(
        visual_index=visual.index,
        dense=dense,
        bm25=bm25,
        config=settings.search.hybrid_temporal,
        visual_dense_ready=True,
        context_dense_ready=context_ready,
        asr_dense_ready=asr_ready,
    )


def _load_dense_temporal(
    settings: AppConfig,
    retrieval: RetrievalService,
    visual: Any,
    messages: list[str],
) -> tuple[DenseTemporalScorer | None, bool, bool]:
    """Load Dense scoring from visual, Context, and projected segment-ASR."""
    context = retrieval.source_retriever(RetrievalSource.CONTEXT)
    asr_retriever = retrieval.source_retriever(RetrievalSource.ASR)
    context_ready = False
    asr_ready = False
    context_index: Any | None = None
    projected_asr: Any | None = None
    text_encoder: Any | None = None

    if context is None:
        messages.append("Dense temporal evidence unavailable: Context retriever missing")
    else:
        try:
            _ = context.index.metadata.embedding_dim
            context_index = context.index
            text_encoder = getattr(context, "encoder", None)
            context_ready = True
        except Exception as error:
            messages.append(
                "Dense temporal evidence identity validation failed: "
                f"{type(error).__name__}: {error}"
            )

    if asr_retriever is None:
        messages.append("Dense temporal evidence unavailable: ASR segment retriever missing")
    else:
        try:
            asr_dimension = asr_retriever.index.metadata.embedding_dim
            if (
                context_ready
                and context_index is not None
                and context_index.metadata.embedding_dim != asr_dimension
            ):
                messages.append(
                    "Dense temporal evidence identity validation failed: "
                    "ValueError: Context and ASR segment index dimensions differ"
                )
                return None, context_ready, False
            projected_asr = SegmentProjectedASRIndex(
                segment_index=asr_retriever.index,
                canonical_index=visual.index,
                projector=asr_retriever.projector,
            )
            if text_encoder is None and getattr(asr_retriever, "encoder", None) is not None:
                text_encoder = asr_retriever.encoder
            asr_ready = True
        except Exception as error:
            messages.append(
                f"Dense temporal ASR projection failed: {type(error).__name__}: {error}"
            )

    try:
        scorer = DenseTemporalScorer(
            visual_index=visual.index,
            context_index=context_index if context_ready else None,
            asr_index=projected_asr if asr_ready else None,
            visual_encoder=visual.encoder,
            text_encoder=text_encoder,
            weights=settings.search.hybrid_temporal.dense,
            chunk_size=settings.search.alignment.chunk_size,
        )
        return scorer, context_ready, asr_ready
    except Exception as error:
        messages.append(
            "Dense temporal evidence identity validation failed: "
            f"{type(error).__name__}: {error}"
        )
        message = str(error)
        if message.startswith("context Dense index identity conflicts"):
            context_ready = False
        elif message.startswith("asr Dense index identity conflicts"):
            asr_ready = False
        else:
            context_ready = False
            asr_ready = False
        return None, context_ready, asr_ready


def _load_bm25_temporal(
    settings: AppConfig,
    visual: Any,
    messages: list[str],
) -> BM25TemporalScorer | None:
    """Load BM25 against the canonical visual-index identity mapping."""
    bm25_path = _runtime_path("HCMAI_BM25_INDEX_PATH", settings.index.bm25_path)
    try:
        return BM25TemporalScorer.load(
            bm25_path,
            visual.index.mapping,
            settings.search.hybrid_temporal.bm25_fields,
        )
    except Exception as error:
        messages.append(
            f"BM25 temporal evidence unavailable at {bm25_path}: "
            f"{type(error).__name__}: {error}"
        )
        return None


def _load_fast_track_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: Any,
    visual_encoder: Any,
    llm: LLMService | None,
    corpus: Corpus | None,
    messages: list[str],
) -> RetrievalService | None:
    """Load optional Context and ASR indexes around the required visual index."""
    if corpus is None:
        messages.append("Canonical frame store unavailable for fast-track retrieval")
        return None

    context = _load_fast_track_index(
        source=RetrievalSource.CONTEXT,
        path=_runtime_path("HCMAI_CONTEXT_INDEX_PATH", settings.index.context_path),
        messages=messages,
    )
    if context is None and RetrievalSource.CONTEXT in settings.search.fusion.required_sources:
        return None

    asr_segment_path = _runtime_path(
        "HCMAI_ASR_SEGMENT_INDEX_PATH",
        settings.index.asr_segment_path,
    )
    asr_segment = _load_fast_track_index(
        source=RetrievalSource.ASR,
        path=asr_segment_path,
        messages=messages,
    )
    if asr_segment is None and RetrievalSource.ASR in settings.search.fusion.required_sources:
        return None

    if (
        context is not None
        and asr_segment is not None
        and context.metadata.embedding_dim != asr_segment.metadata.embedding_dim
    ):
        messages.append(
            "Could not load asr segment index "
            f"{asr_segment_path}: ValueError: embedding dimension differs "
            "from Context index"
        )
        asr_segment = None
        if RetrievalSource.ASR in settings.search.fusion.required_sources:
            return None

    text_encoder = None
    sample = context or asr_segment
    if sample is not None:
        try:
            # Context and ASR share the hosted BGE text family. Constructing
            # this once also guarantees the two retrievers share one cache key.
            text_encoder = _query_encoder(
                models.resolved_evidence_embedding,
                sample,
                llm,
                "text",
            )
        except Exception as error:
            required_text = settings.search.fusion.required_sources.intersection(
                {RetrievalSource.CONTEXT, RetrievalSource.ASR}
            )
            if required_text:
                messages.append(
                    "Could not configure required fast-track text retrieval: "
                    f"{type(error).__name__}: {error}"
                )
                return None
            messages.append(
                "Fast-track text retrieval unavailable; continuing visual-only: "
                f"{type(error).__name__}: {error}"
            )
            context = None
            asr_segment = None

    return RetrievalService.from_fast_track_indexes(
        visual_index=visual,
        visual_encoder=visual_encoder,
        context_index=context,
        asr_segment_index=asr_segment,
        text_encoder=text_encoder,
        corpus=corpus,
        fusion=settings.search.fusion,
        cache_config=settings.search.cache,
        max_projection_gap_ms=settings.index.asr_projection_max_gap_ms,
    )


def _load_fast_track_index(
    *,
    source: RetrievalSource,
    path: Path,
    messages: list[str],
) -> Any | None:
    """Load one optional evidence index, deferring compatibility to its consumer."""
    if not path.is_dir():
        messages.append(f"{source.value.upper()} index not available at {path}")
        return None
    try:
        if source is RetrievalSource.CONTEXT:
            return RetrievalService.load_index(path)
        return SegmentDenseIndex.load(path)
    except Exception as error:
        label = "asr segment" if source is RetrievalSource.ASR else source.value
        messages.append(
            f"Could not load {label} index {path}: {type(error).__name__}: {error}"
        )
        return None


def _query_encoder(
    config: Any,
    index: Any,
    embedding_client: Any = None,
    source: str = "text",
) -> Any:
    """Build a local or configured remote query encoder for one index."""
    if not isinstance(embedding_client, EmbeddingClient):
        try:
            embedding_client = EmbeddingClient(load_embedding_endpoint())
        except Exception:
            embedding_client = None

    if embedding_client is None:
        return EmbeddingService.create_text_adapter(config)
    return EmbeddingService.create_remote_adapter(
        embedding_client,
        config,
        index.metadata.embedding_dim,
        source,
    )


def _runtime_path(environment_name: str, default: str | Path) -> Path:
    """Resolve a retrieval artifact path from its environment override or config."""
    return resolve_repository_path(os.getenv(environment_name, str(default)))
