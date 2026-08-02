"""Public operator-triggered query-suggestion route."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hcmai.common.schemas import QuerySuggestionRequest, QuerySuggestionResponse
from hcmai.common.utils.logging import get_logger

logger = get_logger(__name__)


def create_query_suggestion_router(
    provider_container: dict[str, Any],
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/api/v1/query-suggestions",
        response_model=QuerySuggestionResponse,
    )
    async def suggest_queries(
        request: QuerySuggestionRequest,
    ) -> QuerySuggestionResponse:
        service = provider_container.get("service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Query-suggestion provider is not configured",
            )
        try:
            return service.suggest(request)
        except Exception as error:
            logger.exception("Query-suggestion request failed")
            detail = (str(error).strip() or type(error).__name__)[:160]
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Query-suggestion provider failed: {detail}",
            ) from error

    return router
