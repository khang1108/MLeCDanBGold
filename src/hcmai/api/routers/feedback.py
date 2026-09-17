"""FastAPI router for transactional KIS feedback, query repair, and refinement endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackOpenResponse,
    FeedbackStateResponse,
    FeedbackTurnRequest,
    FeedbackUndoRequest,
)
from hcmai.common.utils.logging import get_logger
from hcmai.kis.feedback.service import FeedbackService
from hcmai.kis.feedback.store import FeedbackError

logger = get_logger(__name__)

_STATUS_BY_CODE: dict[str, int] = {
    "SNAPSHOT_NOT_FOUND": 404,
    "RESULT_NOT_FOUND": 404,
    "SESSION_NOT_FOUND": 404,
    "FEEDBACK_SESSION_NOT_FOUND": 404,
    "STALE_REVISION": 409,
    "FEEDBACK_REVISION_MISMATCH": 409,
    "KIS_REVISION_MISMATCH": 409,
    "TRAIL_REVISION_MISMATCH": 409,
    "REQUEST_CONFLICT": 409,
    "CONSTRAINT_CONFLICT": 409,
    "SNAPSHOT_EXPIRED": 410,
    "FOREIGN_SNAPSHOT_REF": 410,
    "SESSION_EXPIRED": 410,
    "FEEDBACK_SESSION_EXPIRED": 410,
    "CANNOT_UNDO": 422,
    "INVALID_REFERENCE": 422,
    "INVALID_ACTION": 422,
    "SERVICE_UNAVAILABLE": 503,
    "FEEDBACK_UNAVAILABLE": 503,
    "FEEDBACK_STORE_BUSY": 503,
}


def _get_feedback_service(service_container: dict[str, Any]) -> FeedbackService:
    fb = service_container.get("feedback_service")
    if fb is not None:
        return fb
    search_service = service_container.get("service")
    if search_service is not None:
        fb = getattr(search_service, "feedback", None)
        if fb is not None:
            return fb
    raise HTTPException(
        status_code=503,
        detail={"code": "FEEDBACK_UNAVAILABLE", "message": "KIS feedback service not available"},
    )


def create_feedback_router(service_container: dict[str, Any]) -> APIRouter:
    """Create router exposing KIS feedback session, turn, and undo endpoints."""
    router = APIRouter(tags=["kis-feedback"])

    async def _handle_open(request: FeedbackOpenRequest) -> FeedbackOpenResponse:
        service = _get_feedback_service(service_container)
        try:
            state = await run_in_threadpool(service.open, request)
            return FeedbackOpenResponse(
                session_id=state.session_id,
                feedback_revision=state.feedback_revision,
                state=state,
            )
        except FeedbackError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

    async def _handle_turn(
        session_id: str, request: FeedbackTurnRequest
    ) -> FeedbackStateResponse:
        service = _get_feedback_service(service_container)
        try:
            return await run_in_threadpool(service.turn, session_id, request)
        except FeedbackError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

    async def _handle_undo(
        session_id: str, request: FeedbackUndoRequest
    ) -> FeedbackStateResponse:
        service = _get_feedback_service(service_container)
        try:
            return await run_in_threadpool(service.undo, session_id, request)
        except FeedbackError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

    async def _handle_get_state(session_id: str) -> FeedbackStateResponse:
        service = _get_feedback_service(service_container)
        try:
            session = await run_in_threadpool(service._store.get, session_id)
            return session.state
        except FeedbackError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

    # Static routes registered first to avoid dynamic path collision
    router.add_api_route(
        "/api/v1/kis/feedback/open",
        _handle_open,
        methods=["POST"],
        response_model=FeedbackOpenResponse,
    )
    router.add_api_route(
        "/kis/feedback/open",
        _handle_open,
        methods=["POST"],
        response_model=FeedbackOpenResponse,
    )

    # Dynamic routes for turn, undo, and state inspection
    router.add_api_route(
        "/api/v1/kis/feedback/{session_id}/turn",
        _handle_turn,
        methods=["POST"],
        response_model=FeedbackStateResponse,
    )
    router.add_api_route(
        "/kis/feedback/{session_id}/turn",
        _handle_turn,
        methods=["POST"],
        response_model=FeedbackStateResponse,
    )

    router.add_api_route(
        "/api/v1/kis/feedback/{session_id}/undo",
        _handle_undo,
        methods=["POST"],
        response_model=FeedbackStateResponse,
    )
    router.add_api_route(
        "/kis/feedback/{session_id}/undo",
        _handle_undo,
        methods=["POST"],
        response_model=FeedbackStateResponse,
    )

    router.add_api_route(
        "/api/v1/kis/feedback/{session_id}",
        _handle_get_state,
        methods=["GET"],
        response_model=FeedbackStateResponse,
    )
    router.add_api_route(
        "/kis/feedback/{session_id}",
        _handle_get_state,
        methods=["GET"],
        response_model=FeedbackStateResponse,
    )

    return router
