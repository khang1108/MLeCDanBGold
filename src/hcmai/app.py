"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search engine
and the Node.js frontend. It loads online models and frame indexes once at
application startup during the lifespan context.

API Endpoints:
    GET  /health                  - System status and loaded dataset metadata.
    POST /api/v1/search           - Execute a natural language frame search.
    GET  /api/v1/frames/{frame_id} - Fetch frame metadata by unique frame_id.

Usage:
    uv run uvicorn hcmai.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.search import SearchRequest, SearchResponse
from hcmai.data import FrameStore
from hcmai.retriever.dense import DenseRetriever
from hcmai.retriever.encoder import DenseEncoder
from hcmai.retriever.index import VisualIndex
from hcmai.search import SearchEngine


def create_app(search_engine: SearchEngine | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        search_engine: Optional pre-configured ``SearchEngine`` instance.
            If ``None``, the engine is initialized during application lifespan
            using ``FrameStore`` and ``DenseRetriever``.

    Returns:
        Configured ``FastAPI`` application instance.
    """
    engine_container: dict[str, SearchEngine | None] = {"engine": search_engine}

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        if engine_container["engine"] is None:
            metadata_path = os.getenv("HCMAI_METADATA_PATH", "data/aic/metadata/frames.parquet")
            index_path = os.getenv("HCMAI_INDEX_PATH", "artifacts/indexes/visual.index")

            store = FrameStore(metadata_path)
            encoder = DenseEncoder(EncoderConfig())
            index = VisualIndex.load(index_path)
            retriever = DenseRetriever(encoder=encoder, index=index)

            engine_container["engine"] = SearchEngine(
                frame_store=store,
                retriever=retriever,
            )

        yield

    app = FastAPI(
        title="HCMAI 2026 Frame Retrieval API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check() -> dict[str, Any]:
        """Return system health status and metadata readiness.

        Returns:
            Dictionary containing health status, frame store status, and total frames.
        """
        engine = engine_container["engine"]
        store_loaded = engine is not None and getattr(engine, "frame_store", None) is not None
        retriever_loaded = engine is not None and getattr(engine, "retriever", None) is not None
        total_frames = 0

        if store_loaded and hasattr(engine.frame_store, "_records"):
            total_frames = len(engine.frame_store._records)

        return {
            "status": "ok",
            "frame_store_loaded": store_loaded,
            "retriever_loaded": retriever_loaded,
            "total_frames": total_frames,
        }

    @app.post("/api/v1/search", response_model=SearchResponse)
    def search_frames(request: SearchRequest) -> SearchResponse:
        """Execute a frame retrieval query and return ranked results.

        Args:
            request: ``SearchRequest`` schema containing query, top_k, and search_mode.

        Returns:
            ``SearchResponse`` schema containing ranked search results and latency.

        Raises:
            HTTPException: If the search engine is not initialized (503).
        """
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "retriever", None) is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search engine or DenseRetriever not initialized",
            )
        return engine.search(request)

    @app.get("/api/v1/frames/{frame_id}", response_model=FrameRecord)
    def get_frame(frame_id: str) -> FrameRecord:
        """Fetch canonical metadata for a single frame by its frame_id.

        Args:
            frame_id: Globally unique frame identifier (e.g., ``"L21_V001_00000090"``).

        Returns:
            Validated ``FrameRecord`` instance.

        Raises:
            HTTPException: If frame store is uninitialized (503) or frame_id not found (404).
        """
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
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e

    return app


app = create_app()
