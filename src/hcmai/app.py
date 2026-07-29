"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search engine
and the Node.js frontend. It loads online models and frame indexes once at
application startup during the lifespan context.
"""

from __future__ import annotations

import os

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.common.config import AppConfig, EncoderConfig
from hcmai.common.schemas import (
    ConversationSession,
    FrameFeedback,
    FrameRecord,
    KISCSearchRequest,
    KISCSearchResponse,
    RetrievalSource,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
    TaskType,
)
from hcmai.data import ASRStore, CaptionStore, FrameStore, OCRStore
from hcmai.agents.kisc import ConversationResolver, KISCAgent
from hcmai.kisc import KiscSessionManager
from hcmai.llm import InferenceClient, RemoteDenseEncoder
from hcmai.llm.config import LLMServiceConfig
from hcmai.reranking import (
    MultimodalReranker,
    RerankerConfig as PipelineRerankerConfig,
)
from hcmai.retriever.caption import CaptionRetriever
from hcmai.retriever.dense import (
    DenseIndex,
    DenseRetriever,
    create_text_encoder,
)
from hcmai.retriever.fusion import RRFFusionRetriever
from hcmai.search import SearchEngine

logger = get_logger(__name__)


class _UnsupportedSearchTaskError(ValueError):
    """Raised when a non-standalone task reaches the standalone router."""


class _UnavailableSearchPipelineError(RuntimeError):
    """Raised when a known standalone task has no executable pipeline."""


class _SearchEngineUnavailableError(RuntimeError):
    """Raised when the shared frame-search pipeline is not ready."""


class _StandaloneSearchRouter:
    """Dispatch standalone task types without leaking routing into pipelines."""

    def __init__(
        self,
        frame_search: Callable[[SearchRequest], SearchResponse],
    ) -> None:
        self._pipelines: dict[
            TaskType, Callable[[SearchRequest], SearchResponse] | None
        ] = {
            TaskType.KIS: frame_search,
            TaskType.VKIS: frame_search,
            TaskType.VQA: None,
            TaskType.TRAKE: None,
        }

    def dispatch(self, request: SearchRequest) -> SearchResponse:
        """Run the selected standalone pipeline when it is available."""

        if request.query_type not in self._pipelines:
            raise _UnsupportedSearchTaskError(
                f"query_type {request.query_type.value!r} is not a "
                "standalone search task"
            )
        pipeline = self._pipelines[request.query_type]
        if pipeline is None:
            raise _UnavailableSearchPipelineError(
                f"pipeline for query_type {request.query_type.value!r} "
                "is not available"
            )
        logger.info(
            "Routing standalone query_type=%s pipeline=frame_search",
            request.query_type.value,
        )
        return pipeline(request)

    def capabilities(self, frame_search_ready: bool) -> dict[str, bool]:
        """Report which standalone task pipelines can currently execute."""

        return {
            task_type.value: (
                pipeline is not None and frame_search_ready
            )
            for task_type, pipeline in self._pipelines.items()
        }


def _configure_backend_logging() -> None:
    """Make hcmai progress logs visible alongside Uvicorn output."""
    level = os.getenv("HCMAI_LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("HCMAI_LOG_FILE") or None
    configure_logging(level, log_file=log_file)
    get_logger("hcmai").setLevel(level)
    logger.info(
        "Backend logging configured level=%s file=%s",
        level,
        log_file or "console-only",
    )


def _fallback_kisc_agent(engine: SearchEngine) -> KISCAgent:
    """Build a usable agent whose resolver deliberately enters safe fallback."""

    def unavailable_provider(_: dict[str, Any]) -> object:
        raise RuntimeError("structured conversation provider not configured")

    return KISCAgent(ConversationResolver(unavailable_provider), engine)


def _default_kisc_agent(engine: SearchEngine) -> KISCAgent:
    client = getattr(engine, "inference_client", None)
    if client is None:
        return _fallback_kisc_agent(engine)
    return KISCAgent(ConversationResolver(client.resolve_conversation), engine)


def _load_settings(messages: list[str]) -> AppConfig:
    """Load the shared YAML config, falling back to typed defaults."""
    config_path = Path(os.getenv("HCMAI_CONFIG_PATH", "configs/baseline.yaml"))
    if not config_path.is_file():
        messages.append(f"Config not found at {config_path}; using defaults")
        return AppConfig()
    try:
        return AppConfig.from_yaml(config_path)
    except Exception as error:
        messages.append(
            f"Could not load config {config_path}: {type(error).__name__}: {error}"
        )
        return AppConfig()


def _load_model_settings(messages: list[str]) -> LLMServiceConfig:
    """Load the single authoritative model-serving configuration."""

    path = Path(os.getenv("HCMAI_LLM_CONFIG", "llm/config.yaml"))
    if not path.is_file():
        messages.append(f"Model config not found at {path}; using defaults")
        return LLMServiceConfig()
    try:
        return LLMServiceConfig.from_yaml(path)
    except Exception as error:
        messages.append(
            f"Could not load model config {path}: "
            f"{type(error).__name__}: {error}"
        )
        return LLMServiceConfig()


def _build_inference_client(settings: AppConfig) -> InferenceClient | None:
    if not settings.inference.enabled:
        return None
    return InferenceClient(
        os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
        settings.inference.timeout_seconds,
    )


def _build_query_encoder(
    settings: AppConfig,
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
    local_config = config.model_copy(
        update={
            "device": settings.inference.local_fallback_device,
            "batch_size": settings.inference.local_fallback_batch_size,
        }
    )
    local = create_text_encoder(local_config)
    if client is None:
        return local
    fallback = local if settings.inference.local_embedding_fallback else None
    return RemoteDenseEncoder(
        client,
        config,
        index.metadata.embedding_dim,
        fallback,
        source=source,
    )


def _build_remote_reranker(
    settings: AppConfig,
    models: LLMServiceConfig,
    store: FrameStore | None,
    client: InferenceClient | None,
) -> MultimodalReranker | None:
    if client is None or store is None or settings.search.rerank_count <= 0:
        return None
    return MultimodalReranker(
        store,
        PipelineRerankerConfig(batch_size=models.reranker.batch_size),
        client.rerank,
        dataset_root=settings.dataset.root,
    )


def _load_evidence_stores(
    settings: AppConfig,
    messages: list[str],
) -> dict[RetrievalSource, Any]:
    """Load available text artifacts without blocking visual search startup."""

    configured = (
        (
            RetrievalSource.CAPTION,
            CaptionStore,
            settings.dataset.enrichment.caption_path,
        ),
        (RetrievalSource.OCR, OCRStore, settings.dataset.enrichment.ocr_path),
        (RetrievalSource.ASR, ASRStore, settings.dataset.enrichment.asr_path),
    )
    stores: dict[RetrievalSource, Any] = {}
    for source, store_type, path in configured:
        if path is None:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            messages.append(f"{source.value.upper()} artifact not available at {path}")
            continue
        try:
            stores[source] = store_type(path)
            logger.info(
                "%sStore loaded path=%s frames=%d",
                source.value.upper(),
                path,
                len(stores[source]),
            )
        except Exception as error:
            messages.append(
                f"Could not load {source.value} artifact {path}: "
                f"{type(error).__name__}: {error}"
            )
    return stores


def _with_caption_retrieval(
    settings: AppConfig,
    models: LLMServiceConfig,
    visual: DenseRetriever,
    visual_index: DenseIndex,
    client: InferenceClient | None,
    messages: list[str],
) -> DenseRetriever | RRFFusionRetriever:
    """Add compatible caption retrieval without breaking visual-only startup."""

    caption_dir = Path(
        os.getenv("HCMAI_CAPTION_INDEX_PATH", str(settings.index.caption_path))
    )
    if not caption_dir.is_dir():
        messages.append(f"Caption index directory not available at {caption_dir}")
        return visual
    if settings.search.fusion.method != "rrf":
        messages.append(
            f"Unsupported fusion method {settings.search.fusion.method!r}; "
            "caption retrieval disabled"
        )
        return visual
    try:
        caption_index = DenseIndex.load(caption_dir)
        if caption_index.metadata.dataset_version != visual_index.metadata.dataset_version:
            raise ValueError("visual and caption index dataset versions differ")
        caption_config = models.caption_embedding
        caption_encoder = _build_query_encoder(
            settings,
            caption_config,
            caption_index,
            client,
            "caption",
        )
        caption = CaptionRetriever(caption_encoder, caption_index)
        logger.info(
            "Caption retrieval enabled path=%s model=%s dimension=%d rrf_k=%d",
            caption_dir,
            caption_config.model_name,
            caption_index.metadata.embedding_dim,
            settings.search.fusion.rrf_k,
        )
        return RRFFusionRetriever(
            [visual, caption],
            settings.search.fusion,
        )
    except Exception as error:
        messages.append(
            f"Could not enable caption retrieval from {caption_dir}: "
            f"{type(error).__name__}: {error}"
        )
        return visual


def _load_default_engine(messages: list[str]) -> SearchEngine:
    """Load available artifacts without preventing the API from starting."""
    settings = _load_settings(messages)
    models = _load_model_settings(messages)
    metadata_path = Path(
        os.getenv("HCMAI_METADATA_PATH", str(settings.dataset.frames_path))
    )
    index_dir = Path(os.getenv("HCMAI_INDEX_PATH", str(settings.index.path)))

    store = None
    if metadata_path.is_file() and metadata_path.stat().st_size > 0:
        try:
            # Initialize Frame Store for the application
            store = FrameStore(metadata_path)
            logger.info(
                "FrameStore loaded path=%s frames=%d", metadata_path, len(store._records)
            )
        except Exception as error:
            messages.append(
                f"Could not load metadata {metadata_path}: "
                f"{type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Metadata not available at {metadata_path}")

    retriever = None
    inference_client = _build_inference_client(settings)
    if index_dir.is_dir():
        try:
            index = DenseIndex.load(index_dir)
            encoder = _build_query_encoder(
                settings,
                models.visual_embedding,
                index,
                inference_client,
                "visual",
            )
            visual = DenseRetriever(encoder=encoder, index=index)
            retriever = _with_caption_retrieval(
                settings,
                models,
                visual,
                index,
                inference_client,
                messages,
            )
        except Exception as error:
            messages.append(
                f"Could not load index {index_dir}: {type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Index directory not available at {index_dir}")

    reranker = _build_remote_reranker(
        settings, models, store, inference_client
    )
    evidence_stores = _load_evidence_stores(settings, messages)
    engine = SearchEngine(
        frame_store=store,
        retriever=retriever,
        reranker=reranker,
        config=settings.model_dump(mode="python"),
        evidence_stores=evidence_stores,
    )
    setattr(engine, "inference_client", inference_client)
    return engine


def create_app(
    search_engine: SearchEngine | None = None,
    session_manager: KiscSessionManager | None = None,
    kisc_agent: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    engine_container: dict[str, Any] = {
        "engine": search_engine,
        "startup_messages": [],
    }
    provider_container = {"kisc_agent": kisc_agent}
    if (
        provider_container["kisc_agent"] is None
        and search_engine is not None
        and getattr(search_engine, "retriever", None) is not None
    ):
        provider_container["kisc_agent"] = _default_kisc_agent(search_engine)
    kisc_manager = session_manager or KiscSessionManager()
    dataset_root = Path(os.getenv("HCMAI_DATASET_ROOT", "data")).resolve()

    def run_frame_search(request: SearchRequest) -> SearchResponse:
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "retriever", None) is None:
            raise _SearchEngineUnavailableError(
                "Search engine or DenseRetriever not initialized"
            )
        return kisc_manager.process_search(request, engine)

    search_router = _StandaloneSearchRouter(run_frame_search)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        _configure_backend_logging()
        logger.info("Backend startup started")
        if engine_container["engine"] is None:
            engine_container["engine"] = _load_default_engine(
                engine_container["startup_messages"]
            )
        engine = engine_container["engine"]
        if (
            provider_container["kisc_agent"] is None
            and engine is not None
            and getattr(engine, "retriever", None) is not None
        ):
            provider_container["kisc_agent"] = _default_kisc_agent(engine)
        logger.info(
            "Backend startup completed search=%s kisc=%s reranker=%s "
            "remote_inference=%s messages=%d",
            getattr(engine, "retriever", None) is not None,
            provider_container["kisc_agent"] is not None,
            getattr(engine, "reranker", None) is not None,
            getattr(engine, "inference_client", None) is not None,
            len(engine_container["startup_messages"]),
        )
        for message in engine_container["startup_messages"]:
            logger.warning("Backend startup note: %s", message)

        try:
            yield
        finally:
            engine = engine_container["engine"]
            client = getattr(engine, "inference_client", None)
            if client is not None:
                client.client.close()
            logger.info("Backend shutdown completed")

    app = FastAPI(
        title="HCMAI 2026 Frame Retrieval API", version="0.1.0", lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            value.strip()
            for value in os.getenv(
                "HCMAI_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if value.strip()
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("Initializing FastAPI application for the backend service.")

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Return system health status and metadata readiness."""
        engine = engine_container["engine"]
        frame_store = getattr(engine, "frame_store", None)
        retriever = getattr(engine, "retriever", None)
        evidence_stores = getattr(engine, "evidence_stores", {})
        store_loaded = frame_store is not None
        retriever_loaded = retriever is not None
        total_frames = (
            len(frame_store._records)
            if store_loaded and hasattr(frame_store, "_records")
            else 0
        )
        return {
            "status": "ok",
            "ready": store_loaded and retriever_loaded,
            "frame_store_loaded": store_loaded,
            "retriever_loaded": retriever_loaded,
            "total_frames": total_frames,
            "evidence_stores": {
                source.value: source in evidence_stores
                for source in (
                    RetrievalSource.CAPTION,
                    RetrievalSource.OCR,
                    RetrievalSource.ASR,
                )
            },
            "capabilities": {
                "search": retriever_loaded,
                "kisc": provider_container["kisc_agent"] is not None,
                "frame_assets": frame_store is not None,
                "query_types": search_router.capabilities(retriever_loaded),
            },
            "startup_messages": engine_container["startup_messages"],
        }

    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        """Dispatch one of the four standalone competition task types."""
        try:
            response = search_router.dispatch(request)
        except _UnsupportedSearchTaskError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except _UnavailableSearchPipelineError as error:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(error),
            ) from error
        except _SearchEngineUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as e:
            logger.warning("API search request failed error=%s", e)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e
        except Exception:
            logger.exception("API search request failed unexpectedly")
            raise
        return response

    @app.post("/api/v1/session", response_model=ConversationSession)
    async def create_session(problem_id: str | None = None) -> ConversationSession:
        """Create a new KISC conversational session."""
        return kisc_manager.create_session(problem_id=problem_id)

    @app.post("/api/v1/kisc/search", response_model=KISCSearchResponse)
    async def search_kisc(request: KISCSearchRequest) -> KISCSearchResponse:
        """Execute one stateless conversation-resolution and search turn."""
        agent = provider_container["kisc_agent"]
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="KISC provider not initialized",
            )
        try:
            response = agent.search(request)
        except Exception:
            logger.exception("API KISC request failed unexpectedly")
            raise
        return response

    @app.get("/api/v1/sessions", response_model=list[str])
    async def list_session_ids() -> list[str]:
        """Return all conversation session IDs in creation order."""
        return kisc_manager.list_session_ids()

    @app.get("/api/v1/session/{session_id}", response_model=ConversationSession)
    async def get_session(session_id: str) -> ConversationSession:
        """Fetch KISC session history and cumulative feedback state."""
        try:
            return kisc_manager.get_session(session_id)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e

    @app.delete(
        "/api/v1/session/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_session(session_id: str) -> None:
        """Delete one KISC conversation session by its exact ID."""
        try:
            kisc_manager.delete_session(session_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @app.post("/api/v1/feedback", response_model=ConversationSession)
    async def update_feedback(
        session_id: str,
        feedback: FrameFeedback,
    ) -> ConversationSession:
        """Update human frame feedback for a session."""
        try:
            return kisc_manager.update_feedback(session_id, feedback)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e

    @app.get("/api/v1/frames/{frame_id}", response_model=FrameRecord)
    async def get_frame(frame_id: str) -> FrameRecord:
        """Fetch canonical metadata for a single frame by frame_id."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame store not loaded",
            )
        try:
            return engine.frame_store.get(frame_id)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e

    def frame_asset(frame_id: str, *, thumbnail: bool) -> FileResponse:
        engine = engine_container["engine"]
        store = getattr(engine, "frame_store", None)
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame store not loaded",
            )
        try:
            frame = store.get(frame_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        value = frame.thumbnail_path if thumbnail else frame.image_path
        if thumbnail and value is None:
            value = frame.image_path
        path = Path(value).expanduser()
        resolved = (
            path.resolve() if path.is_absolute() else (dataset_root / path).resolve()
        )
        if not resolved.is_relative_to(dataset_root) or not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Frame asset not available",
            )
        return FileResponse(resolved)

    @app.get("/api/v1/frames/{frame_id}/thumbnail")
    async def get_frame_thumbnail(frame_id: str) -> FileResponse:
        """Serve a frame thumbnail without exposing arbitrary file paths."""
        return frame_asset(frame_id, thumbnail=True)

    @app.get("/api/v1/frames/{frame_id}/image")
    async def get_frame_image(frame_id: str) -> FileResponse:
        """Serve a full frame without exposing arbitrary file paths."""
        return frame_asset(frame_id, thumbnail=False)

    @app.get("/api/v1/frames/{frame_id}/neighbors", response_model=list[FrameRecord])
    async def get_frame_neighbors(
        frame_id: str,
        window_ms: int = Query(default=5_000, ge=0, le=60_000),
    ) -> list[FrameRecord]:
        """Fetch same-video frames in a symmetric timestamp window."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame store not loaded",
            )
        try:
            return engine.frame_store.get_neighbors(
                frame_id,
                window_ms=window_ms,
                include_self=True,
            )
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e

    @app.post("/api/v1/submit", response_model=SubmissionResult)
    async def submit_frame(frame_id: str) -> SubmissionResult:
        """Format a selected frame ID into official BTC competition submission code."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame store not loaded",
            )
        try:
            return kisc_manager.format_submission(frame_id, engine.frame_store)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
            ) from e

    return app


app = create_app()
