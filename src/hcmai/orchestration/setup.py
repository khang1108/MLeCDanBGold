"""Load configured artifacts and assemble the online search service once."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.retrieval.embedding.pipeline import EmbeddingService
from hcmai.llm.pipeline import LLMService, LLMServiceConfig
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.reranking.pipeline import RerankerConfig, RerankingService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

logger = get_logger(__name__)

RetrievalProfile = Literal["context_asr_segment", "legacy_specialists"]
_RETRIEVAL_PROFILES: tuple[RetrievalProfile, ...] = (
    "context_asr_segment",
    "legacy_specialists",
)


def load_search_service(messages: list[str]) -> SearchService:
    """Build the single configured pipeline, preserving degraded startup."""
    settings = _load_app_config()
    models = _load_model_config()
    profile_value = os.getenv("HCMAI_RETRIEVAL_PROFILE", settings.index.profile)
    if profile_value not in _RETRIEVAL_PROFILES:
        raise ValueError(
            "HCMAI_RETRIEVAL_PROFILE must be one of "
            "context_asr_segment or legacy_specialists; "
            f"got {profile_value!r}"
        )
    profile = cast(RetrievalProfile, profile_value)
    metadata_path = Path(os.getenv(
        "HCMAI_METADATA_PATH", str(settings.dataset.frames_path)
    ))
    dataset_root = Path(os.getenv(
        "HCMAI_DATASET_ROOT", str(settings.dataset.root)
    ))
    index_dir = Path(os.getenv("HCMAI_INDEX_PATH", str(settings.index.path)))
    data = _load_data(
        settings,
        metadata_path,
        dataset_root,
        messages,
        profile=profile,
    )
    llm = _load_remote_llm(settings, messages)
    retrieval = _load_retrieval(
        settings,
        models,
        index_dir,
        llm,
        messages,
        profile=profile,
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
    path = Path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


def _load_model_config() -> LLMServiceConfig:
    path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found at {path}")
    return LLMServiceConfig.from_yaml(path)


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
    *,
    profile: RetrievalProfile,
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
    if profile == "legacy_specialists":
        _load_legacy_evidence(data, settings, messages)
    else:
        _load_fast_track_data(
            data,
            settings,
            metadata_path,
            dataset_root,
            messages,
        )
    logger.info("DataService loaded path=%s frames=%d", metadata_path, len(data))
    return data


def _load_legacy_evidence(
    data: DataService,
    settings: AppConfig,
    messages: list[str],
) -> None:
    """Attach rollback specialist stores without changing legacy semantics."""

    paths = {
        RetrievalSource.CAPTION: settings.dataset.enrichment.caption_path,
        RetrievalSource.OCR: settings.dataset.enrichment.ocr_path,
        RetrievalSource.ASR: settings.dataset.enrichment.asr_path,
    }
    for source, path in paths.items():
        if path is None:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        try:
            data.load_evidence(source, path)
        except Exception as error:
            messages.append(
                f"Could not load {source.value} artifact {path}: "
                f"{type(error).__name__}: {error}"
            )


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

    context_path = settings.dataset.enrichment.context_path
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

    transcript_path = settings.dataset.enrichment.transcripts_path
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
    *,
    profile: RetrievalProfile,
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
    if profile == "context_asr_segment":
        return _load_fast_track_retrieval(
            settings,
            models,
            visual,
            visual_encoder,
            llm,
            data,
            messages,
        )
    return _load_legacy_retrieval(
        settings,
        models,
        visual,
        visual_encoder,
        llm,
        messages,
    )


def _load_legacy_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: Any,
    visual_encoder: Any,
    llm: LLMService | None,
    messages: list[str],
) -> RetrievalService | None:
    """Compose the rollback specialist indexes through their existing path."""

    text_indexes = _load_text_indexes(
        settings,
        visual,
        models.caption_embedding.model_name,
        messages,
    )
    if text_indexes is None:
        return None
    if not text_indexes:
        return RetrievalService.from_index(
            visual,
            visual_encoder,
            cache_config=settings.search.cache,
        )
    sample = next(iter(text_indexes.values()))
    try:
        text_encoder = _query_encoder(
            models.caption_embedding,
            sample,
            llm,
            "text",
        )
        return RetrievalService.from_indexes(
            visual,
            visual_encoder,
            text_indexes,
            text_encoder,
            settings.search.fusion,
            settings.search.cache,
        )
    except Exception as error:
        if set(text_indexes).intersection(settings.search.fusion.required_sources):
            messages.append(
                "Could not configure required text retrieval: "
                f"{type(error).__name__}: {error}"
            )
            return None
        messages.append(
            "Text retrieval unavailable; continuing visual-only: "
            f"{type(error).__name__}: {error}"
        )
        return RetrievalService.from_index(
            visual,
            visual_encoder,
            cache_config=settings.search.cache,
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

    context_path = Path(os.getenv(
        "HCMAI_CONTEXT_INDEX_PATH", str(settings.index.context_path)
    ))
    asr_segment_path = Path(os.getenv(
        "HCMAI_ASR_SEGMENT_INDEX_PATH", str(settings.index.asr_segment_path)
    ))
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


def _load_text_indexes(
    settings: AppConfig,
    visual: Any,
    expected_model_name: str,
    messages: list[str],
) -> dict[RetrievalSource, Any] | None:
    loaded: dict[RetrievalSource, Any] = {}
    expected_dimension: int | None = None
    for source, path in _text_index_paths(settings).items():
        required = source in settings.search.fusion.required_sources
        if not path.is_dir():
            messages.append(f"{source.value.upper()} index not available at {path}")
            if required:
                return None
            continue
        try:
            index = RetrievalService.load_index(
                path,
                subset_search_threshold=settings.index.subset_search_threshold,
            )
            if index.metadata.dataset_version != visual.metadata.dataset_version:
                raise ValueError("dataset version differs from visual index")
            if index.metadata.model_name != expected_model_name:
                raise ValueError("model differs from configured text encoder")
            if expected_dimension is None:
                expected_dimension = index.metadata.embedding_dim
            elif index.metadata.embedding_dim != expected_dimension:
                raise ValueError("embedding dimension differs from text indexes")
            loaded[source] = index
        except Exception as error:
            messages.append(
                f"Could not load {source.value} index {path}: "
                f"{type(error).__name__}: {error}"
            )
            if required:
                return None
    return loaded


def _text_index_paths(settings: AppConfig) -> dict[RetrievalSource, Path]:
    return {
        RetrievalSource.CAPTION: Path(os.getenv(
            "HCMAI_CAPTION_INDEX_PATH", str(settings.index.caption_path)
        )),
        RetrievalSource.OCR: Path(os.getenv(
            "HCMAI_OCR_INDEX_PATH", str(settings.index.ocr_path)
        )),
        RetrievalSource.ASR: Path(os.getenv(
            "HCMAI_ASR_INDEX_PATH", str(settings.index.asr_path)
        )),
    }


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
