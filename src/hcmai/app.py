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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hcmai.bootstrap import load_default_engine
from hcmai.common.schemas import SearchRequest, SearchResponse
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.orchestration import SearchEngine
from hcmai.routers import (
    StandaloneSearchDispatcher,
    create_frames_router,
    create_query_suggestion_router,
    create_search_router,
    create_system_router,
)
from hcmai.routers.search import SearchEngineUnavailableError

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


def create_app(
    search_engine: SearchEngine | None = None,
    query_suggestion_service: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    engine_container: dict[str, Any] = {
        "engine": search_engine,
        "startup_messages": [],
    }
    suggestion_container = {
        "service": query_suggestion_service
        or getattr(search_engine, "query_suggestion_service", None)
    }
    dataset_root = Path(os.getenv("HCMAI_DATASET_ROOT", "data")).resolve()

    def run_frame_search(request: SearchRequest) -> SearchResponse:
        engine = engine_container["engine"]
        if engine is None or getattr(engine, "retriever", None) is None:
            raise SearchEngineUnavailableError(
                "Search engine or DenseRetriever not initialized"
            )
        return engine.search(request)

    search_dispatcher = StandaloneSearchDispatcher(run_frame_search)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        _configure_backend_logging()
        logger.info("Backend startup started")
        if engine_container["engine"] is None:
            engine_container["engine"] = load_default_engine(
                engine_container["startup_messages"]
            )
        engine = engine_container["engine"]
        if suggestion_container["service"] is None:
            suggestion_container["service"] = getattr(
                engine, "query_suggestion_service", None
            )
        logger.info(
            "Backend startup completed search=%s reranker=%s "
            "remote_inference=%s messages=%d",
            getattr(engine, "retriever", None) is not None,
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
            service = suggestion_container["service"]
            provider = getattr(service, "provider", None)
            close = getattr(provider, "close", None)
            if close is not None:
                close()
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

    app.include_router(
        create_system_router(
            engine_container,
            search_dispatcher,
            suggestion_container,
        )
    )
    app.include_router(create_search_router(search_dispatcher))
    app.include_router(create_query_suggestion_router(suggestion_container))
    app.include_router(create_frames_router(engine_container, dataset_root))

    return app


app = create_app()
