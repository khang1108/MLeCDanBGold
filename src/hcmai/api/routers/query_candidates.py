"""Routing for explicit stateless query-candidate generation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from hcmai.api.contracts.query_candidates import (
    QueryCandidatesRequest,
    QueryCandidatesResponse,
)
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.query_preparation.service import QueryPreparationError


def create_query_candidates_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the query-candidate router over the shared SearchService."""

    router = APIRouter()

    @router.post(
        "/api/v1/query-candidates",
        response_model=QueryCandidatesResponse,
    )
    async def generate_query_candidates(
        request: QueryCandidatesRequest,
    ) -> QueryCandidatesResponse:
        """Resolve event boundaries and generate one stateless response."""

        service = service_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            return await run_in_threadpool(service.generate_query_candidates, request)
        except (SearchServiceUnavailableError, QueryPreparationError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error

    return router