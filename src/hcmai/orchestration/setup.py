"""Read configuration and assemble the online search service once.

This module chooses the startup sequence and combines already-focused corpus,
retrieval, inference, and request-workflow capabilities. It does not load
artifacts or indexes itself.
"""

from __future__ import annotations

import os
from time import monotonic

from hcmai.common.config import (
    AppConfig,
    resolve_dataset_root,
    resolve_repository_path,
)
from hcmai.common.environment import load_repository_environment
from hcmai.common.utils.logging import get_logger
from hcmai.inference import LLMClient, load_llm_endpoint
from hcmai.kis.assets import KISImageAssetStore
from hcmai.kis.resolver import KISIntentResolver
from hcmai.orchestration.corpus_setup import load_corpus
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.retrieval_setup import (
    load_image_encoder,
    load_retrieval,
    load_temporal_evidence,
    select_visual_retriever,
)
from hcmai.retrieval.evidence.literal import LiteralTextIndex
from hcmai.retrieval.translation.service import EventTranslator
# pyrefly: ignore [missing-import]
from llm.config import LLMServiceConfig
# pyrefly: ignore [missing-import]
from llm.pipeline import LLMService

logger = get_logger(__name__)


def load_search_service(messages: list[str]) -> SearchService:
    """Compose the configured online service while retaining startup diagnostics."""
    # Re-apply repository values here so stale terminal exports cannot redirect
    # runtime paths before any data or model capability is constructed.
    load_repository_environment()

    settings = _load_app_config()
    models = _load_model_config()
    if os.getenv("HCMAI_RETRIEVAL_PROFILE") is not None:
        raise ValueError(
            "HCMAI_RETRIEVAL_PROFILE is no longer supported; "
            "use the context/asr-segment runtime artifacts"
        )

    metadata_path = resolve_repository_path(
        os.getenv("HCMAI_METADATA_PATH", str(settings.dataset.frames_path))
    )
    configured_dataset_root = os.getenv("HCMAI_DATASET_ROOT", str(settings.dataset.root))
    dataset_root = resolve_dataset_root(configured_dataset_root)
    configured_dataset_path = resolve_repository_path(configured_dataset_root)
    if dataset_root != configured_dataset_path:
        messages.append(
            "Migrated legacy HCMAI_DATASET_ROOT from "
            f"{configured_dataset_path} to {dataset_root}"
        )

    corpus = load_corpus(settings, metadata_path, dataset_root, messages)
    llm = _load_remote_llm(settings, messages)
    llm_client = _load_llm_client(messages)
    event_translator = _load_event_translator(settings, messages, llm=llm_client)
    intent_resolver = _load_intent_resolver(messages, llm=llm_client)
    retrieval = load_retrieval(settings, models, llm, messages, corpus=corpus)
    visual_retriever = select_visual_retriever(retrieval)
    image_encoder = load_image_encoder(models, visual_retriever, llm, messages)
    temporal_evidence = load_temporal_evidence(
        settings,
        retrieval,
        visual_retriever,
        messages,
    )
    literal_text = LiteralTextIndex(corpus) if corpus is not None else None
    if literal_text is not None:
        logger.info(
            "Literal filter loaded frames=%d sources=%s",
            len(corpus),
            ",".join(literal_text.available_sources) or "none",
        )

    kis_image_assets = _load_kis_image_assets(settings, messages)

    return SearchService(
        corpus=corpus,
        retrieval=retrieval,
        config=settings.search,
        llm=llm,
        event_translator=event_translator,
        temporal_evidence=temporal_evidence,
        image_encoder=image_encoder,
        api_config=settings.api,
        literal_text=literal_text,
        visual_retriever=visual_retriever,
        intent_resolver=intent_resolver,
        kis_image_assets=kis_image_assets,
    )


def _load_app_config() -> AppConfig:
    """Load the repository-owned application configuration."""
    path = resolve_repository_path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


def _load_model_config() -> LLMServiceConfig:
    """Load the configuration for legacy remote model capabilities."""
    path = resolve_repository_path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found at {path}")
    return LLMServiceConfig.from_yaml(path)


def _load_remote_llm(
    settings: AppConfig,
    messages: list[str],
) -> LLMService | None:
    """Create the remaining private inference service and record readiness failures."""
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


def _load_llm_client(messages: list[str]) -> LLMClient | None:
    """Construct the shared LLM client for intent resolution and translation."""
    try:
        return LLMClient(load_llm_endpoint())
    except Exception as error:
        messages.append(f"Remote LLM client unavailable ({error})")
        return None


def _load_intent_resolver(
    messages: list[str],
    llm: LLMClient | None = None,
) -> KISIntentResolver | None:
    """Construct KIS intent resolver using the shared LLM client."""
    if llm is None:
        messages.append("KIS intent resolver unavailable: LLM client not configured")
        return None
    return KISIntentResolver(llm)


def _load_event_translator(
    settings: AppConfig,
    messages: list[str],
    llm: LLMClient | None = None,
) -> EventTranslator | None:
    """Construct event translation using the provider-agnostic LLM client."""
    if llm is None:
        messages.append("Event translation unavailable: LLM client not configured")
        return None
    return EventTranslator(llm, settings.event_translation)


def _load_kis_image_assets(
    settings: AppConfig,
    messages: list[str],
) -> KISImageAssetStore | None:
    """Create the persistent KIS query image asset store.

    Returns None only when storage initialisation fails; the store directory
    is created on first write so a missing directory at startup is not an error.
    """
    try:
        storage_dir = resolve_repository_path(settings.api.kis_query_asset_dir)
        store = KISImageAssetStore(
            storage_dir,
            max_upload_bytes=settings.api.image_max_upload_bytes,
            max_pixels=settings.api.image_max_pixels,
        )
        logger.info("KIS image asset store ready at %s", storage_dir)
        return store
    except Exception as error:
        messages.append(f"KIS image asset store unavailable ({error})")
        return None
