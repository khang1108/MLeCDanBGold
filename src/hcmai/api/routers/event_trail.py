"""FastAPI router for EventTrail session exploration endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts.event_trail import (
    EventTrailActionRequest,
    EventTrailOpenRequest,
    EventTrailStateResponse,
)
from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.service import EventTrailService

_STATUS_BY_CODE: dict[str, int] = {
    "SNAPSHOT_NOT_FOUND": 404,
    "RESULT_NOT_FOUND": 404,
    "TRAIL_SESSION_NOT_FOUND": 404,
    "KIS_REVISION_MISMATCH": 409,
    "KIS_REVISION_CONFLICT": 409,
    "TRAIL_REVISION_CONFLICT": 409,
    "CONSTRAINT_CONFLICT": 409,
    "SNAPSHOT_EXPIRED": 410,
    "TRAIL_SESSION_EXPIRED": 410,
    "INVALID_EVENT": 422,
    "UNKNOWN_EVENT": 422,
    "INVALID_FRAME": 422,
    "FRAME_NOT_FOUND": 422,
    "INVALID_WINDOW": 422,
    "NOTHING_TO_UNDO": 422,
    "CANNOT_UNDO": 422,
    "EVENT_TRAIL_UNAVAILABLE": 503,
}


def _get_event_trail_service(service_container: dict[str, Any]) -> EventTrailService:
    service = service_container.get("service")
    if service is None or getattr(service, "event_trail", None) is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "EVENT_TRAIL_UNAVAILABLE", "message": "EventTrail is unavailable"},
        )
    return service.event_trail


def create_event_trail_router(service_container: dict[str, Any]) -> APIRouter:
    """Create router binding EventTrail exploration endpoints to the service container."""
    router = APIRouter(prefix="/api/v1/event-trail", tags=["event-trail"])

    @router.post("/open", response_model=EventTrailStateResponse)
    async def open_trail(request: EventTrailOpenRequest) -> EventTrailStateResponse:
        service = _get_event_trail_service(service_container)
        try:
            view = await run_in_threadpool(
                service.open,
                request.snapshot_id,
                request.result_id,
                request.expected_kis_revision,
            )
            return EventTrailStateResponse.from_domain(view)
        except EventTrailError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc

    @router.get("/{session_id}", response_model=EventTrailStateResponse)
    async def get_trail_state(session_id: str) -> EventTrailStateResponse:
        service = _get_event_trail_service(service_container)
        try:
            view = await run_in_threadpool(service.get, session_id)
            return EventTrailStateResponse.from_domain(view)
        except EventTrailError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc

    @router.post("/{session_id}/actions", response_model=EventTrailStateResponse)
    async def apply_trail_action(
        session_id: str, request: EventTrailActionRequest
    ) -> EventTrailStateResponse:
        service = _get_event_trail_service(service_container)
        domain_action = request.to_domain_action()
        try:
            view = await run_in_threadpool(
                service.act,
                session_id,
                request.expected_trail_revision,
                domain_action,
            )
            return EventTrailStateResponse.from_domain(view)
        except EventTrailError as exc:
            status = _STATUS_BY_CODE.get(exc.code, 500)
            raise HTTPException(
                status_code=status,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc

    return router
