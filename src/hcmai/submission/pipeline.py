"""Public facade for optional DRES mini-challenge operations."""

from __future__ import annotations

import asyncio

from hcmai.common.schemas import (
    FrameRecord,
    MiniChallengeAnswer,
    MiniChallengeAnswerSet,
    MiniChallengeEvaluation,
    MiniChallengeLoginRequest,
    MiniChallengeLoginResponse,
    MiniChallengeSubmission,
    MiniChallengeSubmissionResult,
    MiniChallengeSubmitRequest,
    MiniChallengeTaskTemplate,
)
from hcmai.common.utils.logging import get_logger
from hcmai.submission.adapters.dres import DRESClient, DRESClientError

logger = get_logger(__name__)


class MiniChallengeService:
    def __init__(self, client: DRESClient) -> None:
        self.client = client
        self._session_id: str | None = None
        self._session_refresh_task: asyncio.Task[None] | None = None

    @classmethod
    def remote(
        cls, base_url: str, *, timeout_seconds: float = 10.0
    ) -> "MiniChallengeService":
        return cls(DRESClient(base_url, timeout_seconds=timeout_seconds))

    async def login(
        self, request: MiniChallengeLoginRequest
    ) -> MiniChallengeLoginResponse:
        response = await self.client.login(request)
        self._session_id = response.session_id
        return response

    @property
    def session_id(self) -> str | None:
        """Return the latest successful login session without exposing credentials."""

        return self._session_id

    async def start_session_refresh(
        self,
        request: MiniChallengeLoginRequest,
        *,
        interval_seconds: float = 300.0,
    ) -> None:
        """Refresh the DRES session immediately and then at a bounded interval."""

        if interval_seconds <= 0:
            raise ValueError("session refresh interval must be positive")
        task = self._session_refresh_task
        if task is not None and not task.done():
            return
        await self._refresh_session_once(request)
        self._session_refresh_task = asyncio.create_task(
            self._refresh_session_loop(request, interval_seconds),
            name="dres-session-refresh",
        )

    async def _refresh_session_loop(
        self,
        request: MiniChallengeLoginRequest,
        interval_seconds: float,
    ) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            await self._refresh_session_once(request)

    async def _refresh_session_once(
        self, request: MiniChallengeLoginRequest
    ) -> None:
        try:
            await self.login(request)
            logger.info("DRES session refreshed successfully")
        except asyncio.CancelledError:
            raise
        except DRESClientError as error:
            logger.warning(
                "DRES session refresh failed status=%d; keeping previous session",
                error.status_code,
            )
        except Exception as error:
            logger.warning(
                "DRES session refresh failed error=%s; keeping previous session",
                type(error).__name__,
            )

    async def list_evaluations(
        self, session: str
    ) -> list[MiniChallengeEvaluation]:
        return await self.client.list_evaluations(session)

    async def current_task(
        self, evaluation_id: str, session: str
    ) -> MiniChallengeTaskTemplate:
        return await self.client.current_task(evaluation_id, session)

    async def submit_frame(
        self,
        evaluation_id: str,
        session: str,
        request: MiniChallengeSubmitRequest,
        frame: FrameRecord,
    ) -> MiniChallengeSubmissionResult:
        if frame.frame_id != request.frame_id:
            raise ValueError("resolved frame identity does not match the request")
        payload = MiniChallengeSubmission(answerSets=[
            MiniChallengeAnswerSet(
                taskName=request.task_name,
                answers=[MiniChallengeAnswer(
                    mediaItemName=frame.video_id,
                    start=frame.timestamp_ms,
                    end=frame.timestamp_ms,
                    text=request.text,
                )],
            )
        ])
        return await self.client.submit(evaluation_id, session, payload)

    async def close(self) -> None:
        task = self._session_refresh_task
        self._session_refresh_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self.client.close()
