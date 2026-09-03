"""Standalone competition-task search routing."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts import (
    FilterRequest,
    FilterResponse,
    SearchRequest,
    SearchResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import SearchServiceUnavailableError

logger = get_logger(__name__)


def create_search_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the standalone frame-search HTTP router."""

    router = APIRouter()

    @router.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        """Validate and delegate one standalone KIS-family search request."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            return await run_in_threadpool(service.search_kis, request)
        except KeyError as error:
            logger.warning("API search request failed error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API search request failed unexpectedly")
            raise

    @router.post("/api/v1/filter", response_model=FilterResponse)
    async def filter_frames(request: FilterRequest) -> FilterResponse:
        """Run direct substring matching over evidence loaded at startup."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        started = perf_counter()
        try:
            response = await run_in_threadpool(service.filter_frames, request)
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        logger.info(
            "Literal filter completed matches=%d page=%d elapsed_ms=%.1f",
            response.total_results,
            response.page_id,
            (perf_counter() - started) * 1_000,
        )
        return response

    return router
