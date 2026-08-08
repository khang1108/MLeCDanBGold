"""Public facade for optional DRES mini-challenge operations."""

from __future__ import annotations

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
from hcmai.submission.adapters.dres import DRESClient


class MiniChallengeService:
    def __init__(self, client: DRESClient) -> None:
        self.client = client

    @classmethod
    def remote(
        cls, base_url: str, *, timeout_seconds: float = 10.0
    ) -> "MiniChallengeService":
        return cls(DRESClient(base_url, timeout_seconds=timeout_seconds))

    async def login(
        self, request: MiniChallengeLoginRequest
    ) -> MiniChallengeLoginResponse:
        return await self.client.login(request)

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
        await self.client.close()
