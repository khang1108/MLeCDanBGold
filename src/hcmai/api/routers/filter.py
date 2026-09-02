"""Thin HTTP transport for exact disk-backed metadata filtering."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts import FilterRequest, FilterResponse
from hcmai.filtering.service import FilterServiceUnavailableError


def _filter_service(container: dict[str, Any]) -> Any:
    """Return Filter service or expose its independent degraded state."""

    startup_error = container.get("filter_error")
    if startup_error is not None:
        raise startup_error
    service = container.get("filter_service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Filter catalog is not available",
        )
    return service


def create_filter_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the Filter route over a standalone FilterService dependency."""

    router = APIRouter()

    @router.post("/api/v1/filter", response_model=FilterResponse)
    async def filter_frames(request: FilterRequest) -> FilterResponse:
        """Validate and execute one bounded exact metadata-filter page."""

        try:
            service = _filter_service(service_container)
            return await run_in_threadpool(service.filter, request)
        except FilterServiceUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

    return router
