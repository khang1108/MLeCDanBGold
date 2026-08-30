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
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from thundercompute.pipeline import LLMService, LLMServiceConfig
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)


def load_search_service(messages: list[str]) -> SearchService:
    """Build explicit KIS/TRAKE workflows while preserving degraded startup."""

    settings = _load_app_config()
    models = _load_model_config()
    if os.getenv("HCMAI_RETRIEVAL_PROFILE") is not None:
        raise ValueError(
            "HCMAI_RETRIEVAL_PROFILE is no longer supported; "
            "use the context/asr-segment runtime artifacts"
        )
    metadata_path = _runtime_path(
        "HCMAI_METADATA_PATH", settings.dataset.frames_path
    )
    configured_dataset_root = os.getenv(
        "HCMAI_DATASET_ROOT", str(settings.dataset.root)
    )
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
    retrieval = _load_retrieval(
        settings,
        models,
        index_dir,
        llm,
        messages,
        corpus=corpus,
    )
    return SearchService(
        corpus=corpus,
        retrieval=retrieval,
        config=settings.search,
        llm=llm,
    )


def _load_app_config() -> AppConfig:
    path = resolve_repository_path(
        os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml")
    )
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


def _load_model_config() -> LLMServiceConfig:
    path = resolve_repository_path(
        os.getenv("HCMAI_LLM_CONFIG", "thundercompute/config.yaml")
    )
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
            "Remote inference readiness unavailable: "
            f"{category or type(error).__name__}"
        )
    return service


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
    except Exception as error:
        messages.append(
            f"Could not load metadata {metadata_path}: "
            f"{type(error).__name__}: {error}"
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
        path = (
            resolve_repository_path(configured_path)
            if configured_path is not None
            else None
        )
        if not _typed_artifact_available(path, allow_directory=False):
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        assert path is not None
        evidence_paths[source] = path

    configured_object_path = enrichment.object_path
    object_path = (
        resolve_repository_path(configured_object_path)
        if configured_object_path is not None
        else None
    )
    if not _typed_artifact_available(object_path, allow_directory=False):
        messages.append(f"OBJECTS artifact not available at {object_path}")
        object_path = None

    configured_transcript_path = enrichment.transcripts_path
    transcript_path = (
        resolve_repository_path(configured_transcript_path)
        if configured_transcript_path is not None
        else None
    )
    if not _typed_artifact_available(transcript_path, allow_directory=True):
        messages.append(
            f"ASR transcript artifact not available at {transcript_path}"
        )
        transcript_path = None

    configured_video_metadata_path = settings.dataset.media_info_path
    video_metadata_path = (
        resolve_repository_path(configured_video_metadata_path)
        if configured_video_metadata_path is not None
        else None
    )
    if not _metadata_directory_available(video_metadata_path):
        messages.append(
            f"VIDEO metadata artifact not available at {video_metadata_path}"
        )
        video_metadata_path = None

    return evidence_paths, object_path, transcript_path, video_metadata_path


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
    return bool(
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
        visual_encoder = _query_encoder(
            models.visual_embedding, visual, llm, "visual"
        )
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


def _load_fast_track_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: Any,
    visual_encoder: Any,
    llm: LLMService | None,
    corpus: Corpus | None,
    messages: list[str],
) -> RetrievalService | None:
    """Load validated retrieval indexes without default reranker wiring.

    The online KIS/TRAKE baseline requires only the visual index. Context and
    segment-ASR indexes remain optional detached retrieval capabilities.
    """

    if corpus is None:
        messages.append(
            "Canonical frame store unavailable for fast-track retrieval"
        )
        return None

    context_path = _runtime_path(
        "HCMAI_CONTEXT_INDEX_PATH", settings.index.context_path
    )
    asr_segment_path = _runtime_path(
        "HCMAI_ASR_SEGMENT_INDEX_PATH", settings.index.asr_segment_path
    )
    encoder_config = models.resolved_evidence_embedding

    context = _load_fast_track_index(
        source=RetrievalSource.CONTEXT,
        path=context_path,
        visual=visual,
        encoder_config=encoder_config,
        settings=settings,
        messages=messages,
    )
    if (
        context is None
        and RetrievalSource.CONTEXT in settings.search.fusion.required_sources
    ):
        return None

    asr_segment = _load_fast_track_index(
        source=RetrievalSource.ASR,
        path=asr_segment_path,
        visual=visual,
        encoder_config=encoder_config,
        settings=settings,
        messages=messages,
    )
    if (
        asr_segment is None
        and RetrievalSource.ASR in settings.search.fusion.required_sources
    ):
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
    visual: Any,
    encoder_config: Any,
    settings: AppConfig,
    messages: list[str],
) -> Any | None:
    """Load one optional v2 evidence index and validate its online contract."""

    if not path.is_dir():
        messages.append(f"{source.value.upper()} index not available at {path}")
        return None
    try:
        if source is RetrievalSource.CONTEXT:
            index = RetrievalService.load_index(path)
            expected_entity_kind = "frame"
        else:
            index = SegmentDenseIndex.load(path)
            expected_entity_kind = "segment"
        metadata = index.metadata
        if metadata.schema_version != "dense-index-v2":
            raise ValueError("metadata must use dense-index-v2")
        if metadata.entity_kind != expected_entity_kind:
            raise ValueError(
                f"entity_kind must be {expected_entity_kind!r}"
            )
        if metadata.retrieval_source != source.value:
            raise ValueError(
                f"retrieval_source must be {source.value!r}"
            )
        if metadata.dataset_version != visual.metadata.dataset_version:
            raise ValueError("dataset version differs from visual index")
        if metadata.model_name != encoder_config.model_name:
            raise ValueError("model differs from configured evidence encoder")
        if metadata.model_revision != encoder_config.revision:
            raise ValueError("revision differs from configured evidence encoder")
        return index
    except Exception as error:
        label = "asr segment" if source is RetrievalSource.ASR else source.value
        messages.append(
            f"Could not load {label} index {path}: "
            f"{type(error).__name__}: {error}"
        )
        return None


def _query_encoder(
    config: Any, index: Any, llm: LLMService | None, source: str
) -> Any:
    if index.metadata.model_name != config.model_name:
        raise ValueError(
            f"{source} index model {index.metadata.model_name!r} does not "
            f"match llm config {config.model_name!r}"
        )
    if llm is None:
        return EmbeddingService.create_text_adapter(config)
    return EmbeddingService.create_remote_adapter(
        llm, config, index.metadata.embedding_dim, source
    )
