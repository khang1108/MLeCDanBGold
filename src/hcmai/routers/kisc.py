"""Conversational KIS session, search, and feedback routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hcmai.common.schemas import (
    ConversationSession,
    FrameFeedback,
    KISCSearchRequest,
    KISCSearchResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.agents.kisc import KiscSessionManager

logger = get_logger(__name__)


def create_kisc_router(
    manager: KiscSessionManager,
    provider_container: dict[str, Any],
) -> APIRouter:
    """Create the in-memory KISC HTTP router."""

    router = APIRouter()

    @router.post("/api/v1/session", response_model=ConversationSession)
    async def create_session(
        problem_id: str | None = None,
    ) -> ConversationSession:
        return manager.create_session(problem_id=problem_id)

    @router.post("/api/v1/kisc/search", response_model=KISCSearchResponse)
    async def search_kisc(request: KISCSearchRequest) -> KISCSearchResponse:
        agent = provider_container["kisc_agent"]
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="KISC provider not initialized",
            )
        try:
            return agent.search(request)
        except Exception:
            logger.exception("API KISC request failed unexpectedly")
            raise

    @router.get("/api/v1/sessions", response_model=list[str])
    async def list_session_ids() -> list[str]:
        return manager.list_session_ids()

    @router.get(
        "/api/v1/session/{session_id}",
        response_model=ConversationSession,
    )
    async def get_session(session_id: str) -> ConversationSession:
        try:
            return manager.get_session(session_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.delete(
        "/api/v1/session/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_session(session_id: str) -> None:
        try:
            manager.delete_session(session_id)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.post("/api/v1/feedback", response_model=ConversationSession)
    async def update_feedback(
        session_id: str,
        feedback: FrameFeedback,
    ) -> ConversationSession:
        try:
            return manager.update_feedback(session_id, feedback)
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    return router
