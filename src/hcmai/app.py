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

from hcmai.common.config import EncoderConfig
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


def create_app(
    search_engine: SearchEngine | None = None,
    session_manager: KiscSessionManager | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    engine_container: dict[str, SearchEngine | None] = {"engine": search_engine}
    kisc_manager = session_manager or KiscSessionManager()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        if engine_container["engine"] is None:
            metadata_path = os.getenv("HCMAI_METADATA_PATH", "data/aic/metadata/frames.parquet")
            index_path = os.getenv("HCMAI_INDEX_PATH", "artifacts/indexes/visual.index")

            store = FrameStore(metadata_path) if Path(metadata_path).is_file() else None
            retriever = None
            if Path(index_path).is_file():
                encoder = DenseEncoder(EncoderConfig())
                index = VisualIndex.load(index_path)
                retriever = DenseRetriever(encoder=encoder, index=index)

            engine_container["engine"] = SearchEngine(frame_store=store, retriever=retriever)

        yield

    app = FastAPI(title="HCMAI 2026 Frame Retrieval API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        """Return system health status and metadata readiness."""
        engine = engine_container["engine"]
        store_loaded = engine is not None and getattr(engine, "frame_store", None) is not None
        retriever_loaded = engine is not None and getattr(engine, "retriever", None) is not None
        total_frames = len(engine.frame_store._records) if store_loaded and hasattr(engine.frame_store, "_records") else 0
        return {
            "status": "ok",
            "frame_store_loaded": store_loaded,
            "retriever_loaded": retriever_loaded,
            "total_frames": total_frames,
        }

    @app.post("/api/v1/search", response_model=SearchResponse)
    def search_frames(request: SearchRequest) -> SearchResponse:
        """Execute a frame retrieval query (supports both standard & KISC turns)."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "retriever", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search engine or DenseRetriever not initialized",
            )
        return kisc_manager.process_search(request, engine)

    @app.post("/api/v1/session", response_model=ConversationSession)
    def create_session(problem_id: str | None = None) -> ConversationSession:
        """Create a new KISC conversational session."""
        return kisc_manager.create_session(problem_id=problem_id)

    @app.get("/api/v1/session/{session_id}", response_model=ConversationSession)
    def get_session(session_id: str) -> ConversationSession:
        """Fetch KISC session history and cumulative feedback state."""
        try:
            return kisc_manager.get_session(session_id)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.post("/api/v1/feedback", response_model=ConversationSession)
    def update_feedback(session_id: str, feedback: FrameFeedback) -> ConversationSession:
        """Update human frame feedback for a session."""
        try:
            return kisc_manager.update_feedback(session_id, feedback)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.get("/api/v1/frames/{frame_id}", response_model=FrameRecord)
    def get_frame(frame_id: str) -> FrameRecord:
        """Fetch canonical metadata for a single frame by frame_id."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Frame store not loaded")
        try:
            return engine.frame_store.get(frame_id)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.get("/api/v1/frames/{frame_id}/neighbors", response_model=list[FrameRecord])
    def get_frame_neighbors(frame_id: str, window: int = Query(default=5, ge=1, le=50)) -> list[FrameRecord]:
        """Fetch +/- N temporal neighbor frames surrounding a target frame."""
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "frame_store", None) is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Frame store not loaded")
        try:
            return engine.frame_store.get_neighbors(frame_id, window=window, include_target=True)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    @app.post("/api/v1/submit", response_model=SubmissionResult)
    def submit_frame(frame_id: str) -> SubmissionResult:
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
