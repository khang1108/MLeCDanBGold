"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search engine
and the Node.js frontend. It loads online models and frame indexes once at
application startup during the lifespan context.
"""

from __future__ import annotations

import os

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.common.config import AppConfig
from hcmai.common.schemas import (
    ConversationSession,
    FrameFeedback,
    FrameRecord,
    KISCSearchRequest,
    KISCSearchResponse,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
    VQARequest,
    VQAResponse,
)
from hcmai.data import FrameStore
from hcmai.agents.kisc import ConversationResolver, KISCAgent
from hcmai.kisc import KiscSessionManager
from hcmai.llm import InferenceClient, RemoteDenseEncoder
from hcmai.reranking import MultimodalReranker
from hcmai.reranking.config import RerankerConfig as PipelineRerankerConfig
from hcmai.retriever.dense import DenseRetriever
from hcmai.retriever.encoder import DenseEncoder
from hcmai.retriever.index import VisualIndex
from hcmai.search import SearchEngine

logger = get_logger(__name__)


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


def _build_query_encoder(
    settings: AppConfig,
    index: VisualIndex,
) -> tuple[Any, InferenceClient | None]:
    local = DenseEncoder(settings.models.embedding)
    if not settings.inference.enabled:
        return local, None
    client = InferenceClient(
        os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
        settings.inference.timeout_seconds,
    )
    fallback = local if settings.inference.local_embedding_fallback else None
    remote = RemoteDenseEncoder(
        client,
        settings.models.embedding,
        index.metadata.embedding_dim,
        fallback,
    )
    return remote, client


def _build_remote_reranker(
    settings: AppConfig,
    store: FrameStore | None,
    client: InferenceClient | None,
) -> MultimodalReranker | None:
    if client is None or store is None or not settings.models.reranker.enabled:
        return None
    return MultimodalReranker(
        store,
        PipelineRerankerConfig(batch_size=settings.models.reranker.batch_size),
        client.rerank,
        dataset_root=settings.dataset.root,
    )


def _load_default_engine(messages: list[str]) -> SearchEngine:
    """Load available artifacts without preventing the API from starting."""
    settings = _load_settings(messages)
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
    inference_client = None
    if index_dir.is_dir():
        try:
            index = VisualIndex.load(index_dir)
            encoder, inference_client = _build_query_encoder(settings, index)
            retriever = DenseRetriever(encoder=encoder, index=index)
        except Exception as error:
            messages.append(
                f"Could not load index {index_dir}: {type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Index directory not available at {index_dir}")

    reranker = _build_remote_reranker(settings, store, inference_client)
    engine = SearchEngine(
        frame_store=store,
        retriever=retriever,
        reranker=reranker,
        config=settings.model_dump(mode="python"),
    )
    setattr(engine, "inference_client", inference_client)
    return engine


def create_app(
    search_engine: SearchEngine | None = None,
    session_manager: KiscSessionManager | None = None,
    kisc_agent: Any | None = None,
    vqa_provider: Callable[[FrameRecord, VQARequest], VQAResponse] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    engine_container: dict[str, Any] = {
        "engine": search_engine,
        "startup_messages": [],
    }
    provider_container = {
        "kisc_agent": kisc_agent,
        "vqa_provider": vqa_provider,
    }
    if (
        provider_container["kisc_agent"] is None
        and search_engine is not None
        and getattr(search_engine, "retriever", None) is not None
    ):
        provider_container["kisc_agent"] = _default_kisc_agent(search_engine)
    kisc_manager = session_manager or KiscSessionManager()
    dataset_root = Path(os.getenv("HCMAI_DATASET_ROOT", "data")).resolve()

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
            "remote_inference=%s vqa=%s messages=%d",
            getattr(engine, "retriever", None) is not None,
            provider_container["kisc_agent"] is not None,
            getattr(engine, "reranker", None) is not None,
            getattr(engine, "inference_client", None) is not None,
            provider_container["vqa_provider"] is not None,
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
            "capabilities": {
                "search": retriever_loaded,
                "kisc": provider_container["kisc_agent"] is not None,
                "vqa": provider_container["vqa_provider"] is not None,
                "frame_assets": frame_store is not None,
            },
            "startup_messages": engine_container["startup_messages"],
        }

    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        """Execute a frame retrieval query (supports both standard & KISC turns)."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "retriever", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search engine or DenseRetriever not initialized",
            )
        try:
            response = kisc_manager.process_search(request, engine)
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

    @app.post("/api/v1/vqa", response_model=VQAResponse)
    async def answer_vqa(request: VQARequest) -> VQAResponse:
        """Answer one question through an injected frame-grounded provider."""
        engine = engine_container["engine"]
        store = getattr(engine, "frame_store", None)
        provider = provider_container["vqa_provider"]
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame store not loaded",
            )
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VQA provider not initialized",
            )
        try:
            frame = store.get(request.frame_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
            ) from error
        try:
            response = provider(frame, request)
            if (
                response.frame_id != request.frame_id
                or response.question != request.question
            ):
                raise ValueError("VQA provider changed request identity")
            return response
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"VQA provider failed: {type(error).__name__}",
            ) from error

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
