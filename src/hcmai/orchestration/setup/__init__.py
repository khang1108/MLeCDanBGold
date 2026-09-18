"""Read configuration and assemble the online search service once.

This module chooses the startup sequence and combines already-focused corpus,
retrieval, inference, and request-workflow capabilities. It does not load
artifacts or indexes itself.
"""

from __future__ import annotations

import os
from time import monotonic
from typing import TYPE_CHECKING

from hcmai.common.config import (
    AppConfig,
    resolve_dataset_root,
    resolve_repository_path,
)
from hcmai.common.environment import load_repository_environment
from hcmai.common.utils.logging import get_logger
from hcmai.inference import LLMClient, load_llm_endpoint
from hcmai.kis.assets import KISImageAssetStore
from hcmai.kis.feedback.resolver import FeedbackResolver
from hcmai.kis.resolution import (
    KISGlobalRewriter,
    KISIntentResolver,
    KISScopedResolver,
)
from hcmai.orchestration.setup.corpus import load_configured_corpus, load_corpus
from hcmai.orchestration.setup.retrieval import (
    load_image_encoder,
    load_retrieval,
    load_temporal_evidence,
    select_visual_retriever,
)
from hcmai.retrieval.evidence.literal import LiteralTextIndex
from hcmai.retrieval.serving.client import RetrievalHttpClient
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.remote import (
    RemoteImageSearchService,
    RemoteTemporalSearchService,
)
# pyrefly: ignore [missing-import]
from llm.config import LLMServiceConfig
# pyrefly: ignore [missing-import]
from llm.pipeline import LLMService

if TYPE_CHECKING:
    from hcmai.orchestration.pipeline import SearchService

logger = get_logger(__name__)


def load_search_service(messages: list[str]) -> SearchService:
    """Compose the configured online service while retaining startup diagnostics."""
    from hcmai.orchestration.pipeline import SearchService
    # Re-apply repository values here so stale terminal exports cannot redirect
    # runtime paths before any data or model capability is constructed.
    load_repository_environment()

    settings = load_app_config()
    if os.getenv("HCMAI_RETRIEVAL_PROFILE") is not None:
        raise ValueError(
            "HCMAI_RETRIEVAL_PROFILE is no longer supported; "
            "use the context/asr-segment runtime artifacts"
        )

    corpus = load_configured_corpus(settings, messages)
    llm_client = _load_llm_client(messages)
    intent_resolver = _load_intent_resolver(messages, llm=llm_client)
    scoped_resolver = _load_scoped_resolver(messages, llm=llm_client)
    global_rewriter = _load_global_rewriter(messages, llm=llm_client)
    feedback_resolver = _load_feedback_resolver(messages, llm=llm_client)
    kis_image_assets = load_kis_image_assets(settings, messages)

    literal_text = LiteralTextIndex(corpus) if corpus is not None else None
    if literal_text is not None and corpus is not None:
        logger.info(
            "Literal filter loaded frames=%d sources=%s",
            len(corpus),
            ",".join(literal_text.available_sources) or "none",
        )

    client_settings = RetrievalClientSettings.from_env()
    client = RetrievalHttpClient(client_settings)
    probe_status = client.probe()
    if not probe_status.ready:
        messages.append(
            f"Remote retrieval service unavailable at {client_settings.target}"
        )

    temporal = RemoteTemporalSearchService(corpus, client) if corpus is not None else None
    image_search = (
        RemoteImageSearchService(
            corpus,
            client,
            max_upload_bytes=settings.api.image_max_upload_bytes,
            max_pixels=settings.api.image_max_pixels,
        )
        if corpus is not None
        else None
    )

    return SearchService(
        corpus=corpus,
        config=settings.search,
        temporal=temporal,
        image_search=image_search,
        remote_retrieval=client,
        api_config=settings.api,
        literal_text=literal_text,
        intent_resolver=intent_resolver,
        scoped_resolver=scoped_resolver,
        global_rewriter=global_rewriter,
        feedback_resolver=feedback_resolver,
        kis_image_assets=kis_image_assets,
    )


def load_app_config() -> AppConfig:
    """Load the repository-owned application configuration."""
    path = resolve_repository_path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


_load_app_config = load_app_config


def load_model_config() -> LLMServiceConfig:
    """Load the configuration for legacy remote model capabilities."""
    path = resolve_repository_path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found at {path}")
    return LLMServiceConfig.from_yaml(path)


_load_model_config = load_model_config


def load_remote_inference(
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


_load_remote_llm = load_remote_inference



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


def _load_scoped_resolver(
    messages: list[str],
    llm: LLMClient | None = None,
) -> KISScopedResolver | None:
    """Construct KIS scoped resolver using the shared LLM client."""
    if llm is None:
        messages.append("KIS scoped resolver unavailable: LLM client not configured")
        return None
    return KISScopedResolver(llm)


def _load_global_rewriter(
    messages: list[str],
    llm: LLMClient | None = None,
) -> KISGlobalRewriter | None:
    """Construct KIS global rewriter using the shared LLM client."""
    if llm is None:
        messages.append("KIS global rewriter unavailable: LLM client not configured")
        return None
    return KISGlobalRewriter(llm)


def _load_feedback_resolver(
    messages: list[str],
    llm: LLMClient | None = None,
) -> FeedbackResolver | None:
    """Construct KIS chat feedback resolver using the shared LLM client."""
    if llm is None:
        messages.append("KIS feedback resolver unavailable: LLM client not configured")
        return None
    return FeedbackResolver(llm)


def load_kis_image_assets(
    settings: AppConfig,
    messages: list[str],
) -> KISImageAssetStore | None:
    """Create the persistent KIS query image asset store.

    Returns None only when storage initialisation fails; the store directory
    is created on first write so a missing directory at startup is not an error.
    """
    try:
        storage_dir = resolve_repository_path(settings.api.kis_query_asset_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
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

_load_kis_image_assets = load_kis_image_assets
