"""FastAPI router for KIS Query Hypothesis lifecycle and mutation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.query_hypothesis import (
    QueryHypothesisMutateRequest,
    QueryHypothesisOpenRequest,
    QueryHypothesisPreviewResponse,
    QueryHypothesisResponse,
    QueryHypothesisUndoRequest,
    to_domain_action,
)
from hcmai.kis.hypothesis.models import QueryHypothesisError
from hcmai.kis.hypothesis.service import QueryHypothesisService

_STATUS_BY_CODE: dict[str, int] = {
    "QUERY_HYPOTHESIS_NOT_FOUND": 404,
    "QUERY_REVISION_CONFLICT": 409,
    "QUERY_HYPOTHESIS_EXPIRED": 410,
    "QUERY_CANNOT_UNDO": 422,
    "INVALID_ACTION": 422,
    "QUERY_HYPOTHESIS_UNAVAILABLE": 503,
}


def _get_query_hypothesis_service(
    service_container: dict[str, Any]
) -> QueryHypothesisService:
    service = service_container.get("service")
    if service is None or getattr(service, "query_hypotheses", None) is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "QUERY_HYPOTHESIS_UNAVAILABLE",
                "message": "Query Hypothesis service is unavailable",
            },
        )
    return service.query_hypotheses


def _map_error(exc: Exception) -> HTTPException:
    if isinstance(exc, QueryHypothesisError):
        status = _STATUS_BY_CODE.get(exc.code, 400)
        return HTTPException(
            status_code=status,
            detail={"code": exc.code, "message": exc.message},
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=422,
            detail={"code": "INVALID_ACTION", "message": str(exc)},
        )
    return HTTPException(
        status_code=500,
        detail={"code": "INTERNAL_ERROR", "message": str(exc)},
    )


def create_query_hypothesis_router(
    service_container: dict[str, Any]
) -> APIRouter:
    """Create router binding Query Hypothesis endpoints to the service container."""
    router = APIRouter(prefix="/api/v1/kis/hypotheses", tags=["query-hypothesis"])

    @router.post("/open", response_model=QueryHypothesisResponse)
    async def open_hypothesis(
        request: QueryHypothesisOpenRequest,
    ) -> QueryHypothesisResponse:
        service = _get_query_hypothesis_service(service_container)
        try:
            view = await run_in_threadpool(
                service.open, request.text, tuple(request.image_refs)
            )
            return QueryHypothesisResponse(
                session_id=view.session_id,
                intent=view.intent,
                query_revision=view.query_revision,
                can_undo=view.can_undo,
            )
        except Exception as exc:
            raise _map_error(exc) from exc

    @router.get("/{session_id}", response_model=QueryHypothesisResponse)
    async def get_hypothesis(session_id: str) -> QueryHypothesisResponse:
        service = _get_query_hypothesis_service(service_container)
        try:
            view = await run_in_threadpool(service.get, session_id)
            return QueryHypothesisResponse(
                session_id=view.session_id,
                intent=view.intent,
                query_revision=view.query_revision,
                can_undo=view.can_undo,
            )
        except Exception as exc:
            raise _map_error(exc) from exc

    @router.post(
        "/{session_id}/preview", response_model=QueryHypothesisPreviewResponse
    )
    async def preview_hypothesis(
        session_id: str, request: QueryHypothesisMutateRequest
    ) -> QueryHypothesisPreviewResponse:
        service = _get_query_hypothesis_service(service_container)
        try:
            action = to_domain_action(request.action)
            preview = await run_in_threadpool(
                service.preview,
                session_id,
                request.expected_query_revision,
                action,
            )
            return QueryHypothesisPreviewResponse(
                base_revision=preview.base_revision,
                intent=preview.intent,
            )
        except Exception as exc:
            raise _map_error(exc) from exc

    @router.post(
        "/{session_id}/commit", response_model=QueryHypothesisResponse
    )
    async def commit_hypothesis(
        session_id: str, request: QueryHypothesisMutateRequest
    ) -> QueryHypothesisResponse:
        service = _get_query_hypothesis_service(service_container)
        try:
            action = to_domain_action(request.action)
            view = await run_in_threadpool(
                service.commit,
                session_id,
                request.expected_query_revision,
                action,
            )
            return QueryHypothesisResponse(
                session_id=view.session_id,
                intent=view.intent,
                query_revision=view.query_revision,
                can_undo=view.can_undo,
            )
        except Exception as exc:
            raise _map_error(exc) from exc

    @router.post("/{session_id}/undo", response_model=QueryHypothesisResponse)
    async def undo_hypothesis(
        session_id: str, request: QueryHypothesisUndoRequest
    ) -> QueryHypothesisResponse:
        service = _get_query_hypothesis_service(service_container)
        try:
            view = await run_in_threadpool(
                service.undo, session_id, request.expected_query_revision
            )
            return QueryHypothesisResponse(
                session_id=view.session_id,
                intent=view.intent,
                query_revision=view.query_revision,
                can_undo=view.can_undo,
            )
        except Exception as exc:
            raise _map_error(exc) from exc

    return router
