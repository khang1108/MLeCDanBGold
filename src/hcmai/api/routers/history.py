"""HTTP history routes and WebSocket submission-file synchronization."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from hcmai.api.contracts import (
    QueryHistoryCreate,
    QueryHistoryList,
    QueryHistoryRecord,
    QueryHistorySubmissionUpdate,
    QueryHistoryViewedFrameUpdate,
    SubmissionFileCreate,
    SubmissionFileDelete,
    SubmissionFileList,
    SubmissionFileUpdate,
    SubmissionFileValidate,
)
from hcmai.api.history import RevisionConflict, WorkspaceStore
from hcmai.orchestration.pipeline import SearchServiceUnavailableError


COMMANDS = {
    "submission_file.create": SubmissionFileCreate,
    "submission_file.update": SubmissionFileUpdate,
    "submission_file.validate": SubmissionFileValidate,
    "submission_file.delete": SubmissionFileDelete,
}


class WorkspaceConnections:
    """Broadcast committed file changes to connected browser clients."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()


    async def connect(self, websocket: WebSocket) -> None:
        """Accept and retain one WebSocket connection."""

        await websocket.accept()
        self.clients.add(websocket)


    def disconnect(self, websocket: WebSocket) -> None:
        """Forget one disconnected client."""

        self.clients.discard(websocket)


    async def broadcast(self, event: dict[str, object]) -> None:
        """Send one committed mutation to every connected client."""

        failed = []
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except RuntimeError:
                failed.append(client)
        for client in failed:
            self.disconnect(client)


def _workspace_store(container: dict[str, Any]) -> WorkspaceStore:
    """Return the configured Workspace store or an HTTP 503."""

    store = container.get("workspace_store")
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workspace storage is not configured",
        )
    return store


def _frame(container: dict[str, Any], frame_id: str) -> Any:
    """Resolve a canonical frame without invoking retrieval."""

    service = container.get("service")
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not initialized",
        )
    try:
        return service.get_frame(frame_id)
    except SearchServiceUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Frame {frame_id!r} not found",
        ) from error


def _validate_snapshot(container: dict[str, Any], data: QueryHistoryCreate) -> None:
    """Verify every replay identity against the canonical frame catalog."""

    snapshot = data.result_snapshot
    keys = [key for key in ("results", "paths") if key in snapshot]
    items = snapshot.get(keys[0]) if len(keys) == 1 else None
    if not isinstance(items, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Result snapshot must contain a results or paths array",
        )
    key = keys[0]

    for item in items:
        frame_ids = item.get("frame_ids") if isinstance(item, dict) else None
        if not isinstance(frame_ids, list) or not all(
            isinstance(frame_id, str) and frame_id.strip()
            for frame_id in frame_ids
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Snapshot frame_ids must be an array of strings",
            )

        if key == "results":
            representative = item.get("frame_id")
            if not isinstance(representative, str) or not representative.strip():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="KIS result must contain frame_id",
                )
            frame_ids = [representative, *frame_ids]

        for frame_id in dict.fromkeys(frame_ids):
            frame = _frame(container, frame_id)
            if key == "paths" and frame.video_id != item.get("video_id"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="TRAKE path contains a frame from another video",
                )


def _allowed_origins() -> set[str]:
    """Read the same browser-origin allowlist used by the HTTP application."""

    return {
        value.strip()
        for value in os.getenv(
            "HCMAI_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if value.strip()
    }


async def _run_file_command(
    store: WorkspaceStore,
    payload: dict[str, Any],
) -> dict[str, object]:
    """Validate and commit one submission-file WebSocket command."""

    model = COMMANDS.get(payload.get("type"))
    if model is None:
        raise ValueError("Unknown submission file command")
    command = model.model_validate(payload)

    if isinstance(command, SubmissionFileCreate):
        file = await run_in_threadpool(
            store.create_submission_file, command.name, command.content
        )
        return {"type": "submission_file.created", "file": file.model_dump()}
    if isinstance(command, SubmissionFileUpdate):
        file = await run_in_threadpool(
            store.update_submission_file,
            command.name,
            command.content,
            command.expected_revision,
        )
        return {"type": "submission_file.updated", "file": file.model_dump()}
    if isinstance(command, SubmissionFileValidate):
        file = await run_in_threadpool(
            store.validate_submission_file,
            command.name,
            command.is_validated,
            command.expected_revision,
        )
        return {"type": "submission_file.updated", "file": file.model_dump()}

    await run_in_threadpool(
        store.delete_submission_file, command.name, command.expected_revision
    )
    return {"type": "submission_file.deleted", "name": command.name}


def create_workspace_router(service_container: dict[str, Any]) -> APIRouter:
    """Create Workspace history, hydration, and synchronization endpoints."""

    router = APIRouter()
    connections = WorkspaceConnections()

    @router.post(
        "/api/v1/query-history",
        response_model=QueryHistoryRecord,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_history(data: QueryHistoryCreate) -> QueryHistoryRecord:
        """Persist one successful KIS or TRAKE result snapshot."""

        _validate_snapshot(service_container, data)
        try:
            return await run_in_threadpool(
                _workspace_store(service_container).create_history, data
            )
        except sqlite3.IntegrityError as error:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Query history ID already exists",
            ) from error

    @router.get("/api/v1/query-history", response_model=QueryHistoryList)
    async def get_recent_history(user_id: str) -> QueryHistoryList:
        """Return one user's newest twenty replay snapshots."""

        items = await run_in_threadpool(
            _workspace_store(service_container).get_recent_history, user_id
        )
        return QueryHistoryList(items=items)

    @router.patch(
        "/api/v1/query-history/{query_id}/viewed-frame",
        response_model=QueryHistoryRecord,
    )
    async def update_viewed_frame(
        query_id: str,
        data: QueryHistoryViewedFrameUpdate,
    ) -> QueryHistoryRecord:
        """Record one canonical frame opened from this result."""

        _frame(service_container, data.frame_id)
        try:
            return await run_in_threadpool(
                _workspace_store(service_container).update_viewed_frame,
                query_id,
                data.frame_id,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail="Query history not found",
            ) from error

    @router.patch(
        "/api/v1/query-history/{query_id}/submission",
        response_model=QueryHistoryRecord,
    )
    async def update_submission(
        query_id: str,
        data: QueryHistorySubmissionUpdate,
    ) -> QueryHistoryRecord:
        """Link a query to submission data already committed to a file."""

        for frame_id in dict.fromkeys(data.frame_ids):
            _frame(service_container, frame_id)
        try:
            return await run_in_threadpool(
                _workspace_store(service_container).update_submission,
                query_id,
                data.submission_file_name,
                data.submission_line,
                data.frame_ids,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get("/api/v1/submission-files", response_model=SubmissionFileList)
    async def list_submission_files() -> SubmissionFileList:
        """Hydrate every shared submission file from SQLite."""

        files = await run_in_threadpool(
            _workspace_store(service_container).list_submission_files
        )
        return SubmissionFileList(files=files)

    @router.websocket("/api/v1/workspace/ws")
    async def workspace_socket(websocket: WebSocket) -> None:
        """Commit file commands and broadcast only committed changes."""

        store = service_container.get("workspace_store")
        origin = websocket.headers.get("origin")
        if store is None:
            await websocket.close(code=1013)
            return
        if origin and origin not in _allowed_origins():
            await websocket.close(code=1008)
            return

        await connections.connect(websocket)
        try:
            while True:
                try:
                    payload = await websocket.receive_json()
                    if not isinstance(payload, dict):
                        raise ValueError("WebSocket command must be a JSON object")
                    event = await _run_file_command(store, payload)
                    await connections.broadcast(event)
                except RevisionConflict as error:
                    await websocket.send_json({
                        "type": "submission_file.conflict",
                        "file": error.file.model_dump(),
                    })
                except (
                    KeyError,
                    ValueError,
                    ValidationError,
                    json.JSONDecodeError,
                    sqlite3.IntegrityError,
                ) as error:
                    await websocket.send_json({
                        "type": "submission_file.error",
                        "message": str(error),
                    })
        except WebSocketDisconnect:
            connections.disconnect(websocket)

    return router


__all__ = ["create_workspace_router"]
