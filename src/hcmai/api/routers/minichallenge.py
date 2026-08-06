"""Thin API proxy for the optional DRES mini-challenge."""

from __future__ import annotations

from typing import Annotated, Any, Never

from fastapi import APIRouter, Header, HTTPException, status

from hcmai.common.schemas import (
    MiniChallengeEvaluation,
    MiniChallengeSubmissionResult,
    MiniChallengeSubmitRequest,
    MiniChallengeTaskTemplate,
)
from hcmai.orchestration.pipeline import SearchServiceUnavailableError
from hcmai.submission.adapters import DRESClientError
from hcmai.submission.pipeline import MiniChallengeService

SessionHeader = Annotated[
    str,
    Header(alias="X-DRES-Session", min_length=1, max_length=4_096),
]


def _service(container: dict[str, Any]) -> MiniChallengeService:
    return container["minichallenge_service"]


def _raise_upstream(error: DRESClientError) -> Never:
    raise HTTPException(status_code=error.status_code, detail=str(error)) from error


def create_minichallenge_router(container: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/v1/minichallenge", tags=["minichallenge"])

    @router.get("/evaluations", response_model=list[MiniChallengeEvaluation])
    async def list_evaluations(
        session: SessionHeader,
    ) -> list[MiniChallengeEvaluation]:
        try:
            return await _service(container).list_evaluations(session)
        except DRESClientError as error:
            _raise_upstream(error)

    @router.get(
        "/evaluations/{evaluation_id}/current-task",
        response_model=MiniChallengeTaskTemplate,
    )
    async def current_task(
        evaluation_id: str,
        session: SessionHeader,
    ) -> MiniChallengeTaskTemplate:
        try:
            return await _service(container).current_task(evaluation_id, session)
        except DRESClientError as error:
            _raise_upstream(error)

    @router.post(
        "/evaluations/{evaluation_id}/submit",
        response_model=MiniChallengeSubmissionResult,
    )
    async def submit(
        evaluation_id: str,
        request: MiniChallengeSubmitRequest,
        session: SessionHeader,
    ) -> MiniChallengeSubmissionResult:
        search = container["service"]
        if search is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search service not initialized",
            )
        try:
            frame = search.get_frame(request.frame_id)
            return await _service(container).submit_frame(
                evaluation_id, session, request, frame
            )
        except SearchServiceUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except DRESClientError as error:
            _raise_upstream(error)

    return router
