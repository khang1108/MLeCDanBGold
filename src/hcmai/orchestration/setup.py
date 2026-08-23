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
from hcmai.data.pipeline import DataService
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.thundercompute.pipeline import LLMService, LLMServiceConfig
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.reranking.pipeline import RerankerConfig, RerankingService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

def load_search_service(messages: list[str]) -> SearchService:
    """Build the single configured pipeline, preserving degraded startup."""
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
    data = _load_data(
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
        data=data,
    )
    reranking = None
    if llm is not None and data is not None and settings.search.rerank_count > 0:
        reranking = RerankingService.remote(
            data,
            RerankerConfig(
                batch_size=models.reranker.batch_size,
                required=settings.search.reranker.required,
            ),
            llm,
            dataset_root=dataset_root,
        )
    return SearchService(
        data=data,
        retrieval=retrieval,
        reranking=reranking,
        config=settings.search,
        llm=llm,
        vqa_config=settings.vqa,
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
        os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml")
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


def _load_data(
    settings: AppConfig,
    metadata_path: Path,
    dataset_root: Path,
    messages: list[str],
) -> DataService | None:
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        messages.append(f"Metadata not available at {metadata_path}")
        return None
    try:
        data = DataService.load(
            metadata_path,
            dataset_root=dataset_root,
        )
    except Exception as error:
        messages.append(
            f"Could not load metadata {metadata_path}: "
            f"{type(error).__name__}: {error}"
        )
        return None
    _load_fast_track_data(
        data,
        settings,
        metadata_path,
        dataset_root,
        messages,
    )
    asset_status = data.frame_asset_status()
    logger.info(
        "DataService loaded path=%s dataset_root=%s frames=%d "
        "frame_assets_ready=%s checked=%d missing=%d",
        metadata_path,
        dataset_root,
        len(data),
        asset_status.ready,
        asset_status.checked,
        asset_status.missing,
    )
    if not asset_status.ready:
        messages.append(
            "Frame assets unavailable under "
            f"{dataset_root}: checked={asset_status.checked} "
            f"missing={asset_status.missing}"
        )
    return data


def _load_fast_track_data(
    data: DataService,
    settings: AppConfig,
    metadata_path: Path,
    dataset_root: Path,
    messages: list[str],
) -> None:
    """Attach typed Context and transcript stores independently when usable.

    Each optional store is validated through :meth:`DataService.load`. A bad
    optional artifact therefore cannot discard canonical frames or a usable
    store from the other evidence family.
    """

    configured_context_path = settings.dataset.enrichment.context_path
    context_path = (
        resolve_repository_path(configured_context_path)
        if configured_context_path is not None
        else None
    )
    if not _typed_artifact_available(context_path, allow_directory=False):
        messages.append(f"CONTEXT artifact not available at {context_path}")
    else:
        assert context_path is not None
        try:
            typed = DataService.load(
                metadata_path,
                dataset_root=dataset_root,
                context_path=context_path,
            )
            data.context_store = typed.context_store
        except Exception as error:
            messages.append(
                f"Could not load context artifact {context_path}: "
                f"{type(error).__name__}: {error}"
            )

    configured_transcript_path = settings.dataset.enrichment.transcripts_path
    transcript_path = (
        resolve_repository_path(configured_transcript_path)
        if configured_transcript_path is not None
        else None
    )
    if not _typed_artifact_available(transcript_path, allow_directory=True):
        messages.append(
            f"ASR transcript artifact not available at {transcript_path}"
        )
    else:
        assert transcript_path is not None
        try:
            typed = DataService.load(
                metadata_path,
                dataset_root=dataset_root,
                transcript_path=transcript_path,
            )
            data.transcript_store = typed.transcript_store
        except Exception as error:
            messages.append(
                f"Could not load transcript artifact {transcript_path}: "
                f"{type(error).__name__}: {error}"
            )


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


def _load_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    index_dir: Path,
    llm: LLMService | None,
    messages: list[str],
    data: DataService | None,
) -> RetrievalService | None:
    if not index_dir.is_dir():
        messages.append(f"Index directory not available at {index_dir}")
        return None
    try:
        visual = RetrievalService.load_index(
            index_dir,
            subset_search_threshold=settings.index.subset_search_threshold,
        )
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
        data,
        messages,
    )


def _load_fast_track_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: Any,
    visual_encoder: Any,
    llm: LLMService | None,
    data: DataService | None,
    messages: list[str],
) -> RetrievalService | None:
    """Load validated Context/ASR bundles and compose the Task 9 factory."""

    if data is None or data.frame_store is None:
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
        frame_store=data.frame_store,
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
            index = RetrievalService.load_index(
                path,
                subset_search_threshold=settings.index.subset_search_threshold,
            )
            expected_entity_kind = "frame"
        else:
            index = SegmentDenseIndex.load(
                path,
                subset_search_threshold=settings.index.subset_search_threshold,
            )
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
