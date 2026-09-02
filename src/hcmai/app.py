"""FastAPI application for the HCMAI frame-retrieval pipeline.

This module exposes the HTTP API boundary between the Python search service
and the Node.js frontend. It loads online models and frame indexes once at
application startup during the lifespan context.
"""

from __future__ import annotations

import os

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.orchestration.pipeline import SearchService
from hcmai.api.routers import (
    create_filter_router,
    create_frames_router,
    create_query_candidates_router,
    create_search_router,
    create_system_router,
    create_trake_router,
)

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
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    service_container: dict[str, Any] = {
        "service": search_service,
        "startup_messages": [],
    }

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
            logger.info("Backend shutdown completed")

    app = FastAPI(
        title="HCMAI 2026 Frame Retrieval API", version="0.1.0", lifespan=lifespan
    )

    @app.middleware("http")
    async def handle_unexpected_errors(
        request: Request, call_next: Any
    ) -> Response:
        """Return a JSON 500 response so CORS can expose backend failures."""
        try:
            return await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled backend request error method=%s path=%s",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
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
    app.include_router(create_query_candidates_router(service_container))
    app.include_router(create_trake_router(service_container))
    app.include_router(create_frames_router(service_container))
    app.include_router(create_filter_router())

    return app


app = create_app()
