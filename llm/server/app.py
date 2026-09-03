"""Application assembly for the private hosted inference API.

Capability endpoints live in focused routers. This module owns only runtime
construction, one-process model lifecycle, and router registration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from llm.pipeline import LLMService
from llm.server.routers import ROUTERS


def create_llm_app(runtime: LLMService | None = None) -> FastAPI:
    """Create an injectable inference API with one process-owned runtime."""

    owned = runtime or LLMService.from_environment()
    app = FastAPI(
        title="HCMAI Private Model Inference",
        version="0.1.0",
        lifespan=_lifespan(owned),
    )
    app.state.runtime = owned
    for router in ROUTERS:
        app.include_router(router)
    return app


def _lifespan(runtime: LLMService):
    """Build the FastAPI lifespan that loads models once per process."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.load()
        try:
            yield
        finally:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()

    return lifespan
