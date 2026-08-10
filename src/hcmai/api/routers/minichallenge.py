"""Thin API proxy for the optional DRES mini-challenge."""

from __future__ import annotations

import os
from typing import Annotated, Any, Never

from fastapi import APIRouter, Header, HTTPException, status

from hcmai.common.schemas import (
    MiniChallengeEvaluation,
    MiniChallengeLoginRequest,
    MiniChallengeLoginResponse,
    MiniChallengeSubmissionResult,
    MiniChallengeSubmitRequest,
    MiniChallengeTaskTemplate,
)
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.submission.adapters import DRESClientError
from hcmai.submission.pipeline import MiniChallengeService

SessionHeader = Annotated[
    str | None,
    Header(alias="X-DRES-Session", max_length=4_096),
]


def _service(container: dict[str, Any]) -> MiniChallengeService:
    return container["minichallenge_service"]


def _raise_upstream(error: DRESClientError) -> Never:
    raise HTTPException(status_code=error.status_code, detail=str(error)) from error


def _environment_session() -> str:
    return (
        os.getenv("DES_SESSION_ID")
        or os.getenv("DRES_SESSION_ID")
        or os.getenv("HCMAI_MINICHALLENGE_SESSION")
        or ""
    ).strip().strip('"').strip("'")


def _resolve_session(service: MiniChallengeService, session: str | None) -> str:
    explicit_session = session.strip() if session and session.strip() else ""
    resolved = (
        service.session_id
        or explicit_session
        or _environment_session()
    )
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail=(
                "Missing session token. Provide X-DRES-Session, set "
                "DES_SESSION_ID, or configure DES_USERNAME and DES_PASSWORD"
            ),
        )
    return resolved


def create_minichallenge_router(container: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/v1/minichallenge", tags=["minichallenge"])

    @router.get("/config")
    async def get_config() -> dict[str, Any]:
        default_session = _service(container).session_id or _environment_session()
        username = (
            os.getenv("DES_USERNAME")
            or os.getenv("DRES_USERNAME")
            or ""
        ).strip().strip('"').strip("'")
        role = (
            os.getenv("DES_ROLE")
            or os.getenv("DRES_ROLE")
            or ""
        ).strip().strip('"').strip("'")
        return {
            "has_default_session": bool(default_session),
            "session_id": default_session if default_session else None,
            "username": username if username else None,
            "role": role if role else None,
        }

    @router.post("/login", response_model=MiniChallengeLoginResponse)
    async def login(
        request: MiniChallengeLoginRequest,
    ) -> MiniChallengeLoginResponse:
        try:
            return await _service(container).login(request)
        except DRESClientError as error:
            _raise_upstream(error)

    @router.get("/evaluations", response_model=list[MiniChallengeEvaluation])
    async def list_evaluations(
        session: SessionHeader = None,
    ) -> list[MiniChallengeEvaluation]:
        resolved_session = _resolve_session(_service(container), session)
        try:
            return await _service(container).list_evaluations(resolved_session)
        except DRESClientError as error:
            _raise_upstream(error)

    @router.get(
        "/evaluations/{evaluation_id}/current-task",
        response_model=MiniChallengeTaskTemplate,
    )
    async def current_task(
        evaluation_id: str,
        session: SessionHeader = None,
    ) -> MiniChallengeTaskTemplate:
        resolved_session = _resolve_session(_service(container), session)
        try:
            return await _service(container).current_task(
                evaluation_id, resolved_session
            )
        except DRESClientError as error:
            _raise_upstream(error)

    @router.post(
        "/evaluations/{evaluation_id}/submit",
        response_model=MiniChallengeSubmissionResult,
    )
    async def submit(
        evaluation_id: str,
        request: MiniChallengeSubmitRequest,
        session: SessionHeader = None,
    ) -> MiniChallengeSubmissionResult:
        resolved_session = _resolve_session(_service(container), session)
        search = container["service"]
        if search is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            frame = search.get_frame(request.frame_id)
            return await _service(container).submit_frame(
                evaluation_id, resolved_session, request, frame
            )
        except SearchServiceUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except DRESClientError as error:
            _raise_upstream(error)

    return router
