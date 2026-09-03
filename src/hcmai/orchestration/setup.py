"""Load configured artifacts and assemble the online search service once."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from typing import Any

from hcmai.common.config import (
    AppConfig,
    resolve_dataset_root,
    resolve_repository_path,
)
from hcmai.common.environment import load_repository_environment

from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.corpus.corpus import _CorpusFrameLoadError
from hcmai.orchestration.pipeline import SearchService
from hcmai.query_preparation.adapters.qwen import QwenQueryPreparationAdapter
from hcmai.query_preparation.service import QueryPreparationService
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.retrieval.evidence.asr_projected import SegmentProjectedASRIndex
from hcmai.retrieval.evidence.bm25 import BM25TemporalScorer
from hcmai.retrieval.evidence.dense import DenseTemporalScorer
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer
from hcmai.retrieval.evidence.literal import LiteralTextIndex
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
# pyrefly: ignore [missing-import]
from llm.config import LLMServiceConfig
# pyrefly: ignore [missing-import]
from llm.pipeline import LLMService

logger = get_logger(__name__)


def load_search_service(messages: list[str]) -> SearchService:
    """Build explicit KIS/TRAKE workflows while preserving degraded startup."""

    # Re-apply the repository .env at this composition boundary so stale
    # terminal/system exports cannot redirect runtime paths.
    load_repository_environment()

    settings = _load_app_config()
    models = _load_model_config()
    if os.getenv("HCMAI_RETRIEVAL_PROFILE") is not None:
        raise ValueError(
            "HCMAI_RETRIEVAL_PROFILE is no longer supported; "
            "use the context/asr-segment runtime artifacts"
        )
    metadata_path = _runtime_path("HCMAI_METADATA_PATH", settings.dataset.frames_path)
    configured_dataset_root = os.getenv("HCMAI_DATASET_ROOT", str(settings.dataset.root))
    dataset_root = resolve_dataset_root(configured_dataset_root)
    configured_dataset_path = resolve_repository_path(configured_dataset_root)
    if dataset_root != configured_dataset_path:
        messages.append(
            "Migrated legacy HCMAI_DATASET_ROOT from "
            f"{configured_dataset_path} to {dataset_root}"
        )
    index_dir = _runtime_path("HCMAI_INDEX_PATH", settings.index.path)
    corpus = _load_corpus(
        settings,
        metadata_path,
        dataset_root,
        messages,
    )
    llm = _load_remote_llm(settings, messages)
    query_preparation = _load_query_preparation(settings, llm, messages)
    retrieval = _load_retrieval(
        settings,
        models,
        index_dir,
        llm,
        messages,
        corpus=corpus,
    )
    image_encoder = _load_image_encoder(models, retrieval, llm, messages)
    temporal_evidence = _load_temporal_evidence(settings, retrieval, messages)
    literal_text = LiteralTextIndex(corpus) if corpus is not None else None
    if literal_text is not None:
        logger.info(
            "Literal filter loaded frames=%d sources=%s",
            len(corpus),
            ",".join(literal_text.available_sources) or "none",
        )
    return SearchService(
        corpus=corpus,
        retrieval=retrieval,
        config=settings.search,
        llm=llm,
        query_preparation=query_preparation,
        temporal_evidence=temporal_evidence,
        image_encoder=image_encoder,
        api_config=settings.api,
        literal_text=literal_text,
    )


def _load_image_encoder(
    models: LLMServiceConfig,
    retrieval: RetrievalService | None,
    llm: LLMService | None,
    messages: list[str],
) -> Any | None:
    """Reuse the local SigLIP2 adapter or bind its hosted image endpoint."""

    if retrieval is None:
        return None
    source_retriever = getattr(retrieval, "source_retriever", None)
    if source_retriever is None:
        return None
    visual = source_retriever(RetrievalSource.VISUAL)
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


def _load_temporal_evidence(
    settings: AppConfig,
    retrieval: RetrievalService | None,
    messages: list[str],
) -> TemporalEvidenceScorer | None:
    """Load independent full-corpus Dense and BM25 temporal capabilities."""

    if retrieval is None:
        return None
    source_retriever = getattr(retrieval, "source_retriever", None)
    if source_retriever is None:
        messages.append("Temporal evidence unavailable: source bindings missing")
        return None
    visual = source_retriever(RetrievalSource.VISUAL)
    if visual is None:
        messages.append("Temporal evidence unavailable: visual Dense index missing")
        return None

    dense, context_ready, asr_ready = _load_dense_temporal(settings, retrieval, visual, messages)
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
    """Load Dense scoring from visual, Context, and projected segment-ASR.

    Segment ASR is loaded by fast-track retrieval. This startup boundary only
    projects its existing index onto the canonical visual frame identities.
    """

    if retrieval is None:
        return None, False, False

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
                f"Dense temporal evidence identity validation failed: {type(error).__name__}: {error}"
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
            else:
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
            asr_ready = False

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
            f"Dense temporal evidence identity validation failed: {type(error).__name__}: {error}"
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


def _load_app_config() -> AppConfig:
    path = resolve_repository_path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


def _load_model_config() -> LLMServiceConfig:
    path = resolve_repository_path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found at {path}")
    return LLMServiceConfig.from_yaml(path)


def _runtime_path(environment_name: str, default: str | Path) -> Path:
    """Resolve an environment override or config path from repository root."""

    return resolve_repository_path(os.getenv(environment_name, str(default)))


def _load_remote_llm(
    settings: AppConfig,
    messages: list[str],
) -> LLMService | None:
    if not settings.inference.enabled:
        return None
    base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
    service = LLMService.remote(base_url, settings.inference)
    try:
        service.readiness(deadline_at=monotonic() + 5.0)
    except Exception as error:
        category = getattr(getattr(error, "category", None), "value", None)
        messages.append(
            "Remote inference readiness unavailable: " f"{category or type(error).__name__}"
        )
    return service


def _load_query_preparation(
    settings: AppConfig,
    llm: LLMService | None,
    messages: list[str],
) -> QueryPreparationService | None:
    """Construct query preparation only when remote readiness advertises it."""

    capabilities = llm.capability_health() if llm is not None else {}
    if not capabilities.get("query_preparation", False):
        messages.append("Query preparation unavailable; Dense search remains available")
        return None
    return QueryPreparationService(
        QwenQueryPreparationAdapter(llm),
        settings.query_preparation,
    )


def _load_corpus(
    settings: AppConfig,
    metadata_path: Path,
    dataset_root: Path,
    messages: list[str],
) -> Corpus | None:
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"Metadata not available at {metadata_path}")
    try:
        evidence_paths, object_path, transcript_path, video_metadata_path = (
            _configured_corpus_artifacts(settings, messages)
        )
        corpus = Corpus.open(
            metadata_path,
            evidence_paths=evidence_paths,
            dataset_root=dataset_root,
            object_counts_path=object_path,
            transcript_path=transcript_path,
            video_metadata_path=video_metadata_path,
        )
    except _CorpusFrameLoadError:
        # Canonical frames are required for identity-preserving retrieval.
        # Unlike optional evidence, malformed or unreadable frame metadata
        # must prevent startup rather than silently disabling search.
        raise
    except Exception as error:
        messages.append(
            f"Could not load metadata {metadata_path}: " f"{type(error).__name__}: {error}"
        )
        return None
    logger.info(
        "Corpus loaded path=%s dataset_root=%s frames=%d",
        metadata_path,
        dataset_root,
        len(corpus),
    )
    return corpus


def _configured_corpus_artifacts(
    settings: AppConfig,
    messages: list[str],
) -> tuple[
    dict[RetrievalSource, Path],
    Path | None,
    Path | None,
    Path | None,
]:
    """Select existing optional Corpus artifacts and retain startup diagnostics.

    ``Corpus.open`` is intentionally the sole runtime loader.  Context and
    raw-detection artifacts are not Corpus inputs, so their availability is
    not used to alter this runtime composition.
    """

    enrichment = settings.dataset.enrichment
    evidence_paths: dict[RetrievalSource, Path] = {}
    for source, configured_path in (
        (RetrievalSource.CAPTION, enrichment.caption_path),
        (RetrievalSource.OCR, enrichment.ocr_path),
    ):
        path = _runtime_optional_path(
            f"HCMAI_{source.value.upper()}_PATH",
            configured_path,
        )
        if not _typed_artifact_available(path, allow_directory=False):
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        assert path is not None
        evidence_paths[source] = path

    configured_object_path = enrichment.object_path
    object_path = _runtime_optional_path(
        "HCMAI_OBJECT_PATH",
        configured_object_path,
    )
    if not _typed_artifact_available(object_path, allow_directory=False):
        messages.append(f"OBJECTS artifact not available at {object_path}")
        object_path = None

    configured_transcript_path = enrichment.transcripts_path
    transcript_path = _runtime_optional_path(
        "HCMAI_TRANSCRIPTS_PATH",
        configured_transcript_path,
    )
    if not _typed_artifact_available(transcript_path, allow_directory=True):
        messages.append(f"ASR transcript artifact not available at {transcript_path}")
        transcript_path = None

    configured_video_metadata_path = settings.dataset.media_info_path
    video_metadata_path = _runtime_optional_path(
        "HCMAI_VIDEO_METADATA_PATH",
        configured_video_metadata_path,
    )
    if not _metadata_directory_available(video_metadata_path):
        messages.append(f"VIDEO metadata artifact not available at {video_metadata_path}")
        video_metadata_path = None

    return evidence_paths, object_path, transcript_path, video_metadata_path


def _runtime_optional_path(
    environment_name: str,
    default: str | Path | None,
) -> Path | None:
    """Resolve an optional artifact path with an explicit env override."""

    configured = os.getenv(environment_name)
    if configured is not None:
        return resolve_repository_path(configured) if configured.strip() else None
    return resolve_repository_path(default) if default is not None else None


def _typed_artifact_available(
    path: Path | None,
    *,
    allow_directory: bool,
) -> bool:
    """Return whether a configured typed artifact contains readable input."""

    if path is None:
        return False
    if path.is_file():
        return path.stat().st_size > 0
    return (
        allow_directory
        and path.is_dir()
        and any(item.is_file() for item in path.rglob("*.parquet"))
    )


def _metadata_directory_available(path: Path | None) -> bool:
    """Return whether an organizer media-info directory has JSON records."""

    return path is not None and path.is_dir() and any(path.glob("*.json"))


def _load_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    index_dir: Path,
    llm: LLMService | None,
    messages: list[str],
    corpus: Corpus | None,
) -> RetrievalService | None:
    if not index_dir.is_dir():
        messages.append(f"Index directory not available at {index_dir}")
        return None
    try:
        visual = RetrievalService.load_index(index_dir)
        visual_encoder = _query_encoder(models.visual_embedding, visual, llm, "visual")
    except Exception as error:
        messages.append(
            f"Could not load required visual index {index_dir}: " f"{type(error).__name__}: {error}"
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


def _load_fast_track_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: Any,
    visual_encoder: Any,
    llm: LLMService | None,
    corpus: Corpus | None,
    messages: list[str],
) -> RetrievalService | None:
    """Load available retrieval indexes without default reranker wiring.

    The online KIS/TRAKE baseline requires only the visual index. Context and
    segment-ASR indexes remain optional detached retrieval capabilities.
    """

    if corpus is None:
        messages.append("Canonical frame store unavailable for fast-track retrieval")
        return None

    context_path = _runtime_path("HCMAI_CONTEXT_INDEX_PATH", settings.index.context_path)
    asr_segment_path = _runtime_path(
        "HCMAI_ASR_SEGMENT_INDEX_PATH", settings.index.asr_segment_path
    )
    encoder_config = models.resolved_evidence_embedding

    context = _load_fast_track_index(
        source=RetrievalSource.CONTEXT,
        path=context_path,
        messages=messages,
    )
    if context is None and RetrievalSource.CONTEXT in settings.search.fusion.required_sources:
        return None

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
                encoder_config,
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
    **kwargs: Any,
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
        messages.append(f"Could not load {label} index {path}: " f"{type(error).__name__}: {error}")
        return None


def _query_encoder(config: Any, index: Any, llm: LLMService | None, source: str) -> Any:
    if llm is None:
        return EmbeddingService.create_text_adapter(config)
    return EmbeddingService.create_remote_adapter(llm, config, index.metadata.embedding_dim, source)
