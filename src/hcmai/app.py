"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search engine
and the Node.js frontend. It loads online models and frame indexes once at
application startup during the lifespan context.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from hcmai.common.config import AppConfig
from hcmai.common.schemas import (
    ConversationSession,
    FrameFeedback,
    FrameRecord,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
)
from hcmai.data import FrameStore
from hcmai.kisc import KiscSessionManager
from hcmai.retriever.dense import DenseRetriever
from hcmai.retriever.encoder import DenseEncoder
from hcmai.retriever.index import VisualIndex
from hcmai.search import SearchEngine


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
            f"Could not load config {config_path}: "
            f"{type(error).__name__}: {error}"
        )
        return AppConfig()


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
            store = FrameStore(metadata_path)
        except Exception as error:
            messages.append(
                f"Could not load metadata {metadata_path}: "
                f"{type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Metadata not available at {metadata_path}")

    retriever = None
    if index_dir.is_dir():
        try:
            index = VisualIndex.load(index_dir)
            encoder = DenseEncoder(settings.models.embedding)
            retriever = DenseRetriever(encoder=encoder, index=index)
        except Exception as error:
            messages.append(
                f"Could not load index {index_dir}: "
                f"{type(error).__name__}: {error}"
            )
    else:
        messages.append(f"Index directory not available at {index_dir}")

    return SearchEngine(
        frame_store=store,
        retriever=retriever,
        config=settings.model_dump(mode="python"),
    )


def create_app(
    search_engine: SearchEngine | None = None,
    session_manager: KiscSessionManager | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    engine_container: dict[str, Any] = {
        "engine": search_engine,
        "startup_messages": [],
    }
    kisc_manager = session_manager or KiscSessionManager()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        if engine_container["engine"] is None:
            engine_container["engine"] = _load_default_engine(
                engine_container["startup_messages"]
            )

        yield

    app = FastAPI(title="HCMAI 2026 Frame Retrieval API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            return kisc_manager.process_search(request, engine)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e

    @app.post("/api/v1/session", response_model=ConversationSession)
    async def create_session(problem_id: str | None = None) -> ConversationSession:
        """Create a new KISC conversational session."""
        return kisc_manager.create_session(problem_id=problem_id)

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
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.post("/api/v1/feedback", response_model=ConversationSession)
    async def update_feedback(
        session_id: str,
        feedback: FrameFeedback,
    ) -> ConversationSession:
        """Update human frame feedback for a session."""
        try:
            return kisc_manager.update_feedback(session_id, feedback)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.get("/api/v1/frames/{frame_id}", response_model=FrameRecord)
    async def get_frame(frame_id: str) -> FrameRecord:
        """Fetch canonical metadata for a single frame by frame_id."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Frame store not loaded")
        try:
            return engine.frame_store.get(frame_id)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.get("/api/v1/frames/{frame_id}/neighbors", response_model=list[FrameRecord])
    async def get_frame_neighbors(
        frame_id: str,
        window_ms: int = Query(default=5_000, ge=0, le=60_000),
    ) -> list[FrameRecord]:
        """Fetch same-video frames in a symmetric timestamp window."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Frame store not loaded")
        try:
            return engine.frame_store.get_neighbors(
                frame_id,
                window_ms=window_ms,
                include_self=True,
            )
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.post("/api/v1/submit", response_model=SubmissionResult)
    async def submit_frame(frame_id: str) -> SubmissionResult:
        """Format a selected frame ID into official BTC competition submission code."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Frame store not loaded")
        try:
            return kisc_manager.format_submission(frame_id, engine.frame_store)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    return app


app = create_app()
