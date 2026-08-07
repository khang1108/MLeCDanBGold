"""Thin HTTP adapter for competition TRAKE."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from hcmai.common.schemas import TRAKERequest, TRAKEResponse
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipeline import (
    SearchPipelineUnavailableError,
    SearchServiceUnavailableError,
    UnsupportedSearchTaskError,
)

logger = get_logger(__name__)


def create_trake_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the TRAKE router over the public SearchService facade."""

    router = APIRouter()

    @router.post("/api/v1/trake", response_model=TRAKEResponse)
    async def align_trake(request: TRAKERequest) -> TRAKEResponse:
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
        except SearchServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except KeyError as error:
            logger.warning("API TRAKE request failed error=%s", error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
        except Exception:
            logger.exception("API TRAKE request failed unexpectedly")
            raise

    return router
