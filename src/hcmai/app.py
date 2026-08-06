"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search service
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

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.orchestration.pipeline import SearchService
from hcmai.api.routers import (
    create_frames_router,
    create_minichallenge_router,
    create_query_suggestion_router,
    create_search_router,
    create_system_router,
)
from hcmai.submission.pipeline import MiniChallengeService

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
    search_service: SearchService | None = None,
    minichallenge_service: MiniChallengeService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    service_container: dict[str, Any] = {
        "service": search_service,
        "startup_messages": [],
        "minichallenge_service": minichallenge_service or MiniChallengeService.remote(
            os.getenv(
                "HCMAI_MINICHALLENGE_BASE_URL",
                "http://if-wan4.selab.edu.vn:20740",
            ),
            timeout_seconds=float(
                os.getenv("HCMAI_MINICHALLENGE_TIMEOUT_SECONDS", "10")
            ),
        ),
    }
    dataset_root = Path(os.getenv("HCMAI_DATASET_ROOT", "data")).resolve()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        _configure_backend_logging()
        logger.info("Backend startup started")
        if service_container["service"] is None:
            service_container["service"] = SearchService.load(
                service_container["startup_messages"]
            )
        service = service_container["service"]
        health = service.health(service_container["startup_messages"])
        logger.info(
            "Backend startup completed search=%s reranker=%s "
            "remote_inference=%s messages=%d",
            health["capabilities"]["search"],
            getattr(service, "reranking", None) is not None,
            service.llm is not None,
            len(service_container["startup_messages"]),
        )
        for message in service_container["startup_messages"]:
            logger.warning("Backend startup note: %s", message)

        try:
            yield
        finally:
            service = service_container["service"]
            close = getattr(service, "close", None)
            if close is not None:
                close()
            await service_container["minichallenge_service"].close()
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
        create_system_router(service_container)
    )
    app.include_router(create_search_router(service_container))
    app.include_router(create_query_suggestion_router(service_container))
    app.include_router(create_minichallenge_router(service_container))
    app.include_router(create_frames_router(service_container, dataset_root))

    return app


app = create_app()
