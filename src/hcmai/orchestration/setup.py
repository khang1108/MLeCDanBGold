"""Load configured artifacts and assemble the online search service once."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic
from typing import Any

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.embedding.pipeline import EmbeddingService
from hcmai.llm.pipeline import LLMService, LLMServiceConfig
from hcmai.orchestration.pipeline import SearchService
from hcmai.reranking.pipeline import RerankerConfig, RerankingService
from hcmai.retriever.pipeline import RetrievalService

logger = get_logger(__name__)


def load_search_service(messages: list[str]) -> SearchService:
    """Build the single configured pipeline, preserving degraded startup."""
    settings = _load_app_config()
    models = _load_model_config()
    metadata_path = Path(os.getenv(
        "HCMAI_METADATA_PATH", str(settings.dataset.frames_path)
    ))
    index_dir = Path(os.getenv("HCMAI_INDEX_PATH", str(settings.index.path)))
    data = _load_data(settings, metadata_path, messages)
    llm = _load_remote_llm(settings, messages)
    retrieval = _load_retrieval(
        settings, models, index_dir, llm, messages
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
            dataset_root=settings.dataset.root,
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
    settings: AppConfig, metadata_path: Path, messages: list[str]
) -> DataService | None:
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        messages.append(f"Metadata not available at {metadata_path}")
        return None
    try:
        data = DataService.load(
            metadata_path,
            dataset_root=settings.dataset.root,
        )
    except Exception as error:
        messages.append(
            f"Could not load metadata {metadata_path}: "
            f"{type(error).__name__}: {error}"
        )
        return None
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
    logger.info("DataService loaded path=%s frames=%d", metadata_path, len(data))
    return data


def _load_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    index_dir: Path,
    llm: LLMService | None,
    messages: list[str],
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
