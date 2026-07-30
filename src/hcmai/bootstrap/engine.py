"""Build the configured online retrieval engine at application startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hcmai.agents.kisc import ConversationResolver, KISCAgent
from hcmai.common.config import AppConfig, EncoderConfig
from hcmai.common.utils.logging import get_logger
from hcmai.data import FrameStore
from hcmai.llm import InferenceClient, RemoteDenseEncoder
from hcmai.llm.config import LLMServiceConfig
from hcmai.orchestration import SearchEngine
from hcmai.reranking import MultimodalReranker
from hcmai.reranking import RerankerConfig as PipelineRerankerConfig
from hcmai.retriever.dense import DenseIndex, DenseRetriever, create_text_encoder
from hcmai.retriever.fusion import RRFFusionRetriever

from .artifacts import load_evidence_stores, load_text_indexes

logger = get_logger(__name__)


def default_kisc_agent(engine: SearchEngine) -> KISCAgent | None:
    """Create the KISC agent only when remote inference is configured."""
    client = getattr(engine, "inference_client", None)
    if client is None:
        return None
    return KISCAgent(ConversationResolver(client.resolve_conversation), engine)


def _load_settings() -> AppConfig:
    path = Path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Config not found at {path}")
    return AppConfig.from_yaml(path)


def _load_model_settings() -> LLMServiceConfig:
    path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found at {path}")
    return LLMServiceConfig.from_yaml(path)


def _build_inference_client(settings: AppConfig) -> InferenceClient | None:
    if not settings.inference.enabled:
        return None
    base_url = os.getenv(
        "HCMAI_INFERENCE_BASE_URL", settings.inference.base_url
    )
    return InferenceClient(base_url, settings.inference.timeout_seconds)


def _build_query_encoder(
    config: EncoderConfig,
    index: DenseIndex,
    client: InferenceClient | None,
    source: str,
) -> Any:
    if index.metadata.model_name != config.model_name:
        raise ValueError(
            f"{source} index model {index.metadata.model_name!r} does not "
            f"match llm config {config.model_name!r}"
        )
    if client is None:
        return create_text_encoder(config)
    return RemoteDenseEncoder(
        client, config, index.metadata.embedding_dim, source=source
    )


def _build_remote_reranker(
    settings: AppConfig,
    models: LLMServiceConfig,
    store: FrameStore | None,
    client: InferenceClient | None,
) -> MultimodalReranker | None:
    if client is None or store is None or settings.search.rerank_count <= 0:
        return None
    config = PipelineRerankerConfig(batch_size=models.reranker.batch_size)
    return MultimodalReranker(
        store, config, client.rerank, dataset_root=settings.dataset.root
    )


def _with_multimodal_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: DenseRetriever,
    visual_index: DenseIndex,
    client: InferenceClient | None,
) -> DenseRetriever | RRFFusionRetriever:
    if settings.search.fusion.method != "rrf":
        raise ValueError(
            f"Unsupported fusion method {settings.search.fusion.method!r}"
        )
    loaded = load_text_indexes(settings, visual_index)
    encoder = _build_query_encoder(
        models.caption_embedding, loaded[0][3], client, "caption"
    )
    retrievers: list[Any] = [visual]
    for source, retriever_type, index_dir, index in loaded:
        retrievers.append(retriever_type(encoder, index))
        logger.info(
            "%s retrieval enabled path=%s model=%s dimension=%d",
            source.value.upper(), index_dir,
            models.caption_embedding.model_name, index.metadata.embedding_dim,
        )
    logger.info(
        "Weighted RRF enabled sources=visual,caption,ocr,asr rrf_k=%d",
        settings.search.fusion.rrf_k,
    )
    return RRFFusionRetriever(retrievers, settings.search.fusion)


def load_default_engine(messages: list[str]) -> SearchEngine:
    """Load configured artifacts and return the online search engine."""
    settings = _load_settings()
    models = _load_model_settings()
    metadata_path = Path(os.getenv(
        "HCMAI_METADATA_PATH", str(settings.dataset.frames_path)
    ))
    index_dir = Path(os.getenv("HCMAI_INDEX_PATH", str(settings.index.path)))

    store = None
    if metadata_path.is_file() and metadata_path.stat().st_size > 0:
        try:
            store = FrameStore(metadata_path)
            logger.info(
                "FrameStore loaded path=%s frames=%d",
                metadata_path, len(store._records),
            )
        except Exception as error:
            messages.append(
                f"Could not load metadata {metadata_path}: "
                f"{type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Metadata not available at {metadata_path}")

    retriever = None
    client = _build_inference_client(settings)
    if index_dir.is_dir():
        try:
            index = DenseIndex.load(index_dir)
            encoder = _build_query_encoder(
                models.visual_embedding, index, client, "visual"
            )
            visual = DenseRetriever(encoder=encoder, index=index)
            retriever = _with_multimodal_retrieval(
                settings, models, visual, index, client
            )
        except Exception as error:
            messages.append(
                f"Could not load index {index_dir}: {type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Index directory not available at {index_dir}")

    reranker = _build_remote_reranker(settings, models, store, client)
    engine = SearchEngine(
        frame_store=store,
        retriever=retriever,
        reranker=reranker,
        config=settings.model_dump(mode="python"),
        evidence_stores=load_evidence_stores(settings, messages),
    )
    setattr(engine, "inference_client", client)
    return engine
