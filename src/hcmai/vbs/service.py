"""Session cache and active evaluation/task resolution for VBS participants.

This service owns participant-to-secret lookup and retry policy. It never
returns DRES session tokens to API callers or stores them in shared state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

from hcmai.vbs.client import (
    DresAuthenticationError,
    DresClient,
    DresError,
    DresNoActiveTaskError,
)
from hcmai.vbs.config import DresSettings
from hcmai.vbs.models import (
    ApiClientAnswer,
    ApiClientSubmission,
    DresEvaluation,
    DresTaskScope,
    DresTaskTemplateInfo,
    QueryResultLog,
)


class DresUnknownUserError(DresError):
    """No backend credential mapping exists for the supplied VBS user ID."""


class DresEvaluationSelectionError(DresError):
    """No unique active evaluation can be selected safely."""


_T = TypeVar("_T")


class DresService:
    """Manage backend-only participant sessions and current DRES task scope."""

    def __init__(self, settings: DresSettings, client: DresClient | None = None) -> None:
        """Create an in-memory session cache over one reusable HTTP client."""

        self.settings = settings
        self.client = client or DresClient(settings)
        self._sessions: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def connect(self, user_id: str) -> dict[str, bool | str]:
        """Log in using the backend credential mapped to this VBS user ID."""

        async with self._lock(user_id):
            if user_id not in self._sessions:
                self._sessions[user_id] = await self._login(user_id)
            return {"user_id": user_id, "connected": True}

    def session_status(self, user_id: str) -> dict[str, bool | str]:
        """Return a browser-safe connection state without returning its token."""

        return {"user_id": user_id, "connected": user_id in self._sessions}

    async def disconnect(self, user_id: str) -> dict[str, bool | str]:
        """Evict only this process's session cache entry for the participant."""

        async with self._lock(user_id):
            self._sessions.pop(user_id, None)
        return {"user_id": user_id, "connected": False}

    async def list_evaluations(self, user_id: str) -> list[DresEvaluation]:
        """List DRES evaluations visible to the participant's active session."""

        return await self._with_session(
            user_id,
            lambda session_id: self.client.list_evaluations(session_id),
        )

    async def resolve_evaluation(self, user_id: str) -> str:
        """Use configured evaluation or require exactly one visible ACTIVE run."""

        if self.settings.evaluation_id:
            return self.settings.evaluation_id
        evaluations = await self.list_evaluations(user_id)
        active = [evaluation for evaluation in evaluations if evaluation.status == "ACTIVE"]
        if len(active) != 1:
            raise DresEvaluationSelectionError(
                "DRES must expose exactly one ACTIVE evaluation when no evaluation ID is configured"
            )
        return active[0].id

    async def resolve_task(
        self,
        user_id: str,
        evaluation_id: str | None = None,
    ) -> DresTaskTemplateInfo:
        """Resolve the official current-task template fields without inventing an ID."""

        selected_evaluation = evaluation_id or await self.resolve_evaluation(user_id)
        task = await self._with_session(
            user_id,
            lambda session_id: self.client.get_current_task(
                selected_evaluation,
                session_id,
            ),
        )
        return task

    async def resolve_scope(self, user_id: str) -> DresTaskScope:
        """Return a deterministic backend scope for the active DRES task template."""

        evaluation_id = await self.resolve_evaluation(user_id)
        task = await self.resolve_task(user_id, evaluation_id)
        return DresTaskScope(
            evaluation_id=evaluation_id,
            task_scope_key=_task_scope_key(evaluation_id, task),
            task_name=task.name,
            task_group=task.task_group,
            task_type=task.task_type,
            duration=task.duration,
        )

    def media_item_name(self, video_id: str) -> str:
        """Map canonical video identity once, with only an exact prefix strip."""

        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("DRES media item name requires a non-blank video_id")
        prefix = self.settings.media_id_prefix_to_strip
        media_item_name = video_id[len(prefix):] if prefix and video_id.startswith(prefix) else video_id
        if not media_item_name.strip():
            raise ValueError("DRES media item name must not be blank")
        return media_item_name

    def temporal_answer(self, video_id: str, timestamp_ms: int) -> ApiClientAnswer:
        """Build a point answer from exact canonical media ID and milliseconds."""

        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a non-negative integer")
        return ApiClientAnswer(
            media_item_name=self.media_item_name(video_id),
            start=timestamp_ms,
            end=timestamp_ms,
        )

    async def submit(
        self,
        user_id: str,
        evaluation_id: str,
        payload: ApiClientSubmission,
    ):
        """Forward one submission through the participant's session."""

        return await self._with_session(
            user_id,
            lambda session_id: self.client.submit(evaluation_id, session_id, payload),
        )

    async def log_results(
        self,
        user_id: str,
        evaluation_id: str,
        payload: QueryResultLog,
    ):
        """Forward one result log through the participant's DRES session."""

        return await self._with_session(
            user_id,
            lambda session_id: self.client.log_results(evaluation_id, session_id, payload),
        )

    async def aclose(self) -> None:
        """Close the HTTP adapter without persisting its in-memory sessions."""

        await self.client.aclose()

    async def _login(self, user_id: str) -> str:
        """Resolve one credential mapping and return its DRES token internally."""

        credential = self.settings.credentials.get(user_id)
        if credential is None:
            raise DresUnknownUserError(
                f"No DRES credential is configured for VBS user ID {user_id!r}"
            )
        user = await self.client.login(
            credential.username,
            credential.password.get_secret_value(),
        )
        return user.session_id

    async def _with_session(
        self,
        user_id: str,
        operation: Callable[[str], Awaitable[_T]],
        *,
        retry_auth: bool = True,
    ) -> _T:
        """Run once with a cached token and retry at most once after a 401."""

        session_id = self._sessions.get(user_id)
        if session_id is None:
            raise DresAuthenticationError("Participant is not connected to DRES")
        try:
            return await operation(session_id)
        except DresAuthenticationError:
            if not retry_auth:
                raise
            async with self._lock(user_id):
                latest = self._sessions.get(user_id)
                if latest == session_id:
                    latest = await self._login(user_id)
                    self._sessions[user_id] = latest
            if latest is None:
                raise DresAuthenticationError("Participant is not connected to DRES") from None
            return await operation(latest)

    def _lock(self, user_id: str) -> asyncio.Lock:
        """Return the lock that serializes login and refresh for one participant."""

        lock = self._locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_id] = lock
        return lock


def _task_scope_key(evaluation_id: str, task: DresTaskTemplateInfo) -> str:
    """Hash the exact official task fields into a versioned private scope key."""

    canonical = json.dumps(
        {
            "evaluation_id": evaluation_id,
            "name": task.name,
            "taskGroup": task.task_group,
            "taskType": task.task_type,
            "duration": task.duration,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"dres-task-v1:{hashlib.sha256(canonical).hexdigest()}"


__all__ = [
    "DresEvaluationSelectionError",
    "DresService",
    "DresUnknownUserError",
]
