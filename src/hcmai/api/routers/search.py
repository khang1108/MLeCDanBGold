"""Standalone competition-task search routing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from hcmai.common.schemas import SearchRequest, SearchResponse, TaskType
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import (
    SearchPipelineUnavailableError,
    SearchServiceUnavailableError,
    UnsupportedSearchTaskError,
)

logger = get_logger(__name__)


def create_search_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the standalone frame-search HTTP router."""

    router = APIRouter()

    @router.post("/api/v1/search", response_model=SearchResponse)
    async def search_frames(request: SearchRequest) -> SearchResponse:
        if request.session_id is not None or request.feedback is not None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "session_id and feedback are not supported by "
                    "standalone search"
                ),
            )
        if request.query_type is TaskType.KISC:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="query_type 'kisc' is not a standalone search task",
            )
        if request.query_type is TaskType.VQA:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="query_type 'vqa' must use /api/v1/vqa",
            )
        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            return await run_in_threadpool(service.search, request)
        except UnsupportedSearchTaskError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error
        except SearchPipelineUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(error),
            ) from error
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

    return router
