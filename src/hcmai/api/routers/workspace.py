"""HTTP hydration and WebSocket collaboration for shared answer candidates.

The router accepts only participant IDs and candidate data. Credential lookup,
DRES sessions, revisions, and durable task scope remain in backend services.
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import TypeAdapter, ValidationError

from hcmai.api.contracts.workspace import (
    AnswerAddFrame,
    AnswerAddText,
    AnswerClear,
    AnswerDelete,
    AnswerModeSet,
    AnswerTaskClearAndSwitch,
    AnswerUpdateFrame,
    AnswerUpdateText,
    AnswerWorkspaceCommand,
    AnswerWorkspaceEvent,
    AnswerWorkspaceSnapshot,
)
from hcmai.api.history import (
    AnswerSubmissionInFlight,
    AnswerWorkspaceConflict,
    AnswerWorkspaceTaskScopeMismatch,
    WorkspaceStore,
)
from hcmai.vbs.client import DresError
from hcmai.vbs.models import DresTaskScope


_COMMAND = TypeAdapter(AnswerWorkspaceCommand)


class WorkspaceSocketHub:
    """Broadcast committed snapshots only to sockets on the same DRES task."""

    def __init__(self) -> None:
        """Start with no connected browser sockets."""

        self._sockets: dict[WebSocket, tuple[str, str]] = {}
        self._hydrating: dict[WebSocket, list[AnswerWorkspaceEvent]] = {}

    def connect(self, socket: WebSocket, evaluation_id: str, task_scope_key: str) -> None:
        """Register one already-hydrated socket for its active DRES scope."""

        self._sockets[socket] = (evaluation_id, task_scope_key)
        self._hydrating.pop(socket, None)

    def begin_hydration(self, socket: WebSocket, evaluation_id: str, task_scope_key: str) -> None:
        """Subscribe before reading the first snapshot and buffer concurrent commits."""

        self._sockets[socket] = (evaluation_id, task_scope_key)
        self._hydrating[socket] = []

    def disconnect(self, socket: WebSocket) -> None:
        """Remove one socket without disturbing other participants."""

        self._sockets.pop(socket, None)
        self._hydrating.pop(socket, None)

    async def broadcast(self, event: AnswerWorkspaceEvent) -> None:
        """Send an event only to matching evaluation/task collaborators."""

        payload = event.model_dump(mode="json")
        event_scope = (
            event.workspace.active_evaluation_id,
            event.workspace.active_task_scope_key,
        )
        recipients = [
            socket
            for socket, socket_scope in tuple(self._sockets.items())
            if socket_scope == event_scope and socket not in self._hydrating
        ]
        for socket, socket_scope in tuple(self._sockets.items()):
            if socket_scope == event_scope and socket in self._hydrating:
                self._hydrating[socket].append(event)
        await self._send(payload, recipients)

    async def complete_hydration(
        self,
        socket: WebSocket,
        initial_snapshot: AnswerWorkspaceSnapshot,
    ) -> None:
        """Replay newer buffered commits before enabling direct broadcasts."""

        last_revision = initial_snapshot.revision
        last_scope = (
            initial_snapshot.active_evaluation_id,
            initial_snapshot.active_task_scope_key,
        )
        while socket in self._hydrating:
            pending = self._hydrating[socket]
            self._hydrating[socket] = []
            if not pending:
                # No await separates this check from marking the socket ready,
                # so the next event either joins the next batch or sends direct.
                self._hydrating.pop(socket, None)
                return
            for event in pending:
                event_scope = (
                    event.workspace.active_evaluation_id,
                    event.workspace.active_task_scope_key,
                )
                is_newer = event.workspace.revision > last_revision
                is_equal_revision_task_switch = (
                    event.workspace.revision == last_revision
                    and event.type == "answer.task.switched"
                    and event_scope != last_scope
                )
                if not is_newer and not is_equal_revision_task_switch:
                    continue
                try:
                    await socket.send_json(event.model_dump(mode="json"))
                except (RuntimeError, WebSocketDisconnect):
                    self.disconnect(socket)
                    return
                last_revision = max(last_revision, event.workspace.revision)
                last_scope = event_scope

    def scope_for(self, socket: WebSocket) -> tuple[str, str] | None:
        """Return the scope captured when one socket completed hydration."""

        return self._sockets.get(socket)

    async def rebind_and_broadcast_task_switch(
        self,
        event: AnswerWorkspaceEvent,
        *,
        old_evaluation_id: str,
        old_task_scope_key: str,
    ) -> None:
        """Move old-scope collaborators to the committed new task and notify both sides.

        Sockets already hydrated against the new live scope are collaborators too.
        Other scopes never receive this event or change their registration.
        """

        if event.type != "answer.task.switched":
            raise ValueError("Only a committed task-switch event can rebind sockets")
        old_scope = (old_evaluation_id, old_task_scope_key)
        new_scope = (
            event.workspace.active_evaluation_id,
            event.workspace.active_task_scope_key,
        )
        transition_scopes = {old_scope, new_scope}
        recipients = [
            socket
            for socket, scope in tuple(self._sockets.items())
            if scope in transition_scopes
        ]
        for socket in recipients:
            self._sockets[socket] = new_scope
        ready_recipients = [socket for socket in recipients if socket not in self._hydrating]
        for socket in recipients:
            if socket in self._hydrating:
                self._hydrating[socket].append(event)
        await self._send(event.model_dump(mode="json"), ready_recipients)

    async def _send(
        self,
        payload: dict[str, Any],
        recipients: list[WebSocket],
    ) -> None:
        """Deliver one snapshot and remove sockets that are already closed."""

        stale: list[WebSocket] = []
        for socket in recipients:
            try:
                await socket.send_json(payload)
            except (RuntimeError, WebSocketDisconnect):
                stale.append(socket)
        for socket in stale:
            self.disconnect(socket)


def create_answer_workspace_router(service_container: dict[str, Any]) -> APIRouter:
    """Create the user-scoped HTTP and collaborative WebSocket endpoints."""

    router = APIRouter()
    socket_hub = WorkspaceSocketHub()

    def _store() -> WorkspaceStore:
        store = service_container.get("workspace_store")
        if store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Answer workspace database is not configured",
            )
        return store

    def _vbs_service() -> Any:
        service = service_container.get("vbs_service")
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DRES session service is not configured",
            )
        return service

    async def _scope(user_id: str) -> DresTaskScope:
        """Require a connected participant and resolve the active DRES scope."""

        service = _vbs_service()
        if not service.session_status(user_id).get("connected"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Connect the VBS participant before opening the answer workspace",
            )
        try:
            return await service.resolve_scope(user_id)
        except DresError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to resolve the active DRES task",
            ) from error

    async def _snapshot(user_id: str) -> AnswerWorkspaceSnapshot:
        scope = await _scope(user_id)
        return _store().get_answer_workspace(
            scope.evaluation_id,
            scope.task_scope_key,
            scope.task_name,
        )

    async def _broadcast(event: AnswerWorkspaceEvent) -> None:
        """Send committed state to connected collaborators, dropping dead sockets."""
        await socket_hub.broadcast(event)

    service_container["broadcast_workspace"] = _broadcast
    service_container["workspace_socket_hub"] = socket_hub

    @router.get(
        "/api/v1/answer-workspace",
        response_model=AnswerWorkspaceSnapshot,
    )
    async def get_answer_workspace(
        user_id: str | None = Header(default=None, alias="X-VBS-User-ID"),
    ) -> AnswerWorkspaceSnapshot:
        """Hydrate the shared durable workspace for a connected participant."""

        if user_id is None or not user_id.strip():
            raise HTTPException(status_code=401, detail="VBS user ID is required")
        return await _snapshot(user_id)

    @router.websocket("/api/v1/answer-workspace/ws")
    async def answer_workspace_socket(websocket: WebSocket) -> None:
        """Hydrate first, then apply revision-checked commands and broadcast commits."""

        user_id = (
            websocket.headers.get("x-vbs-user-id", "").strip()
            or websocket.query_params.get("user_id", "").strip()
        )
        if not user_id:
            await websocket.close(code=1008, reason="VBS user ID is required")
            return
        origin = websocket.headers.get("origin")
        if origin is not None and not _origin_allowed(origin):
            await websocket.close(code=1008, reason="Browser origin is not allowed")
            return
        try:
            scope = await _scope(user_id)
        except HTTPException as error:
            await websocket.close(code=1008, reason=str(error.detail))
            return
        await websocket.accept()
        socket_hub.begin_hydration(
            websocket,
            scope.evaluation_id,
            scope.task_scope_key,
        )
        try:
            try:
                snapshot = _store().get_answer_workspace(
                    scope.evaluation_id,
                    scope.task_scope_key,
                    scope.task_name,
                )
            except HTTPException as error:
                await websocket.close(code=1011, reason=str(error.detail))
                return
            await websocket.send_json(
                AnswerWorkspaceEvent(
                    type="answer.workspace.hydrated",
                    workspace=snapshot,
                ).model_dump(mode="json")
            )
            await socket_hub.complete_hydration(websocket, snapshot)
            while True:
                raw = await websocket.receive_json()
                await _dispatch_command(websocket, raw, user_id)
        except WebSocketDisconnect:
            pass
        finally:
            socket_hub.disconnect(websocket)

    async def _dispatch_command(websocket: WebSocket, raw: Any, user_id: str) -> None:
        """Validate one client command and broadcast only a durable commit."""

        try:
            command = _COMMAND.validate_python(raw)
            scope = await _scope(user_id)
            if isinstance(command, AnswerTaskClearAndSwitch):
                allowed_scopes = {
                    (command.expected_old_evaluation_id, command.expected_old_task_scope_key),
                    (scope.evaluation_id, scope.task_scope_key),
                }
                if socket_hub.scope_for(websocket) not in allowed_scopes:
                    await websocket.send_json({
                        "type": "answer.error",
                        "code": "TASK_SCOPE_MISMATCH",
                        "message": "This socket is not connected to the workspace being switched",
                    })
                    return
            store = _store()
            event_type = await _apply_command(
                store,
                service_container,
                command,
                user_id,
                scope,
            )
            event = AnswerWorkspaceEvent(
                type=event_type,
                workspace=store.get_answer_workspace(
                    scope.evaluation_id,
                    scope.task_scope_key,
                    scope.task_name,
                ),
            )
            if isinstance(command, AnswerTaskClearAndSwitch):
                await socket_hub.rebind_and_broadcast_task_switch(
                    event,
                    old_evaluation_id=command.expected_old_evaluation_id,
                    old_task_scope_key=command.expected_old_task_scope_key,
                )
            else:
                await _broadcast(event)
        except ValidationError as error:
            await websocket.send_json({
                "type": "answer.error",
                "code": "INVALID_COMMAND",
                "message": "Answer workspace command is invalid",
            })
        except AnswerWorkspaceConflict:
            await _send_conflict(websocket, user_id)
        except AnswerWorkspaceTaskScopeMismatch as error:
            await websocket.send_json({
                "type": "answer.error",
                "code": "TASK_SCOPE_MISMATCH",
                "message": str(error),
                "workspace": (await _snapshot(user_id)).model_dump(mode="json"),
            })
        except AnswerSubmissionInFlight:
            await websocket.send_json({
                "type": "answer.error",
                "code": "SUBMISSION_IN_FLIGHT",
                "message": "A DRES submission is still being resolved",
                "workspace": (await _snapshot(user_id)).model_dump(mode="json"),
            })
        except HTTPException as error:
            await websocket.send_json({
                "type": "answer.error",
                "code": "WORKSPACE_UNAVAILABLE",
                "message": str(error.detail),
            })

    async def _send_conflict(websocket: WebSocket, user_id: str) -> None:
        """Return the latest state to one stale client without changing storage."""

        try:
            latest = await _snapshot(user_id)
        except HTTPException as error:
            await websocket.send_json({
                "type": "answer.error",
                "code": "WORKSPACE_UNAVAILABLE",
                "message": str(error.detail),
            })
            return
        await websocket.send_json({
            "type": "answer.conflict",
            "workspace": latest.model_dump(mode="json"),
        })

    return router


async def _apply_command(
    store: WorkspaceStore,
    container: dict[str, Any],
    command: AnswerWorkspaceCommand,
    user_id: str,
    scope: DresTaskScope,
) -> str:
    """Apply one typed workspace command under its freshly resolved task scope."""

    if isinstance(command, AnswerAddFrame):
        if command.source_frame_id is not None:
            service = container.get("service")
            if service is None:
                raise HTTPException(503, "Canonical frame store is unavailable")
            try:
                frame = service.get_frame(command.source_frame_id)
            except KeyError as error:
                raise HTTPException(404, "Source frame no longer exists") from error
            if frame.video_id != command.video_id or frame.timestamp_ms != command.timestamp_ms:
                raise HTTPException(409, "Frame answer must preserve its exact canonical timestamp")
        store.add_frame_candidate(
            video_id=command.video_id,
            timestamp_ms=command.timestamp_ms,
            source_frame_id=command.source_frame_id,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
            expected_workspace_revision=command.expected_workspace_revision,
        )
        return "answer.added"
    if isinstance(command, AnswerAddText):
        store.add_text_candidate(
            text=command.text,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
            expected_workspace_revision=command.expected_workspace_revision,
        )
        return "answer.added"
    if isinstance(command, AnswerUpdateFrame):
        store.update_frame_candidate(
            candidate_id=command.candidate_id,
            video_id=command.video_id,
            timestamp_ms=command.timestamp_ms,
            expected_candidate_revision=command.expected_candidate_revision,
            expected_workspace_revision=command.expected_workspace_revision,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
        )
        return "answer.updated"
    if isinstance(command, AnswerUpdateText):
        store.update_text_candidate(
            candidate_id=command.candidate_id,
            text=command.text,
            expected_candidate_revision=command.expected_candidate_revision,
            expected_workspace_revision=command.expected_workspace_revision,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
        )
        return "answer.updated"
    if isinstance(command, AnswerDelete):
        store.delete_candidate(
            candidate_id=command.candidate_id,
            expected_candidate_revision=command.expected_candidate_revision,
            expected_workspace_revision=command.expected_workspace_revision,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
        )
        return "answer.deleted"
    if isinstance(command, AnswerClear):
        store.clear_answer_candidates(
            expected_workspace_revision=command.expected_workspace_revision,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
        )
        return "answer.cleared"
    if isinstance(command, AnswerModeSet):
        store.set_avs_enabled(
            avs_enabled=command.avs_enabled,
            expected_workspace_revision=command.expected_workspace_revision,
            user_id=user_id,
            evaluation_id=scope.evaluation_id,
            task_scope_key=scope.task_scope_key,
            task_name=scope.task_name,
        )
        return "answer.mode.changed"
    if isinstance(command, AnswerTaskClearAndSwitch):
        if (
            command.target_evaluation_id,
            command.target_task_scope_key,
            command.target_task_name,
        ) != (
            scope.evaluation_id,
            scope.task_scope_key,
            scope.task_name,
        ):
            raise HTTPException(409, "Target task is not the currently active DRES task")
        store.clear_and_switch_task(
            expected_workspace_revision=command.expected_workspace_revision,
            expected_old_evaluation_id=command.expected_old_evaluation_id,
            expected_old_task_scope_key=command.expected_old_task_scope_key,
            target_evaluation_id=scope.evaluation_id,
            target_task_scope_key=scope.task_scope_key,
            target_task_name=scope.task_name,
            user_id=user_id,
        )
        return "answer.task.switched"
    raise HTTPException(422, "Unknown workspace command")


def _origin_allowed(origin: str) -> bool:
    """Apply the same configured browser-origin allowlist to WebSockets."""

    allowed = {
        value.strip()
        for value in os.getenv(
            "HCMAI_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,https://hcmai.iamphuckhang.dev,https://hcmai.iamphuckhnag.dev",
        ).split(",")
        if value.strip()
    }
    origin_regex = os.getenv(
        "HCMAI_CORS_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1|(.+\.)?iamphuckhang\.dev|(.+\.)?iamphuckhnag\.dev)(:\d+)?$",
    )
    return origin in allowed or re.match(origin_regex, origin) is not None


__all__ = ["WorkspaceSocketHub", "create_answer_workspace_router"]
