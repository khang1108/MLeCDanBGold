"""Integration tests for SQLite Workspace history and submission files."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hcmai.api.contracts import QueryHistoryCreate
from hcmai.api.history import RevisionConflict, WorkspaceStore
from hcmai.app import create_app
from hcmai.corpus.models import Frame


FRAME_A = "L21_V001_00000090"
FRAME_B = "L21_V001_00000120"


class FrameService:
    """Expose canonical frame lookup without retrieval for Workspace tests."""

    llm = None
    reranking = None

    def __init__(self) -> None:
        self.lookups: list[str] = []
        self.frames = {
            frame_id: Frame(
                frame_id=frame_id,
                video_id="L21_V001",
                frame_idx=index,
                timestamp_ms=index * 40,
                image_path=f"/{frame_id}.jpg",
            )
            for index, frame_id in enumerate((FRAME_A, FRAME_B), start=90)
        }


    def get_frame(self, frame_id: str) -> Frame:
        """Return one known canonical frame."""

        self.lookups.append(frame_id)
        if frame_id not in self.frames:
            raise KeyError(frame_id)
        return self.frames[frame_id]


    def health(self, messages: list[str]) -> dict[str, Any]:
        """Return the fields read by application startup logging."""

        del messages
        return {
            "capabilities": {"search": False},
            "remote_inference": {},
        }


    def close(self) -> None:
        """Provide the lifecycle hook used by the application."""


@pytest.fixture(autouse=True)
def inline_workspace_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep SQLite calls inline for deterministic ASGI tests."""

    async def run_inline(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr("hcmai.api.routers.history.run_in_threadpool", run_inline)


@pytest.fixture
def workspace_store(tmp_path: Path) -> WorkspaceStore:
    """Create an isolated Workspace database."""

    return WorkspaceStore(tmp_path / "workspace.sqlite3")


@pytest.fixture
def workspace_app(workspace_store: WorkspaceStore) -> FastAPI:
    """Create an app with canonical frame lookup and Workspace storage."""

    return create_app(
        search_service=FrameService(),
        workspace_store=workspace_store,
    )


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Send one request through the ASGI application."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def history_payload(index: int, *, user_id: str = "user-a") -> dict[str, Any]:
    """Build one complete KIS response snapshot."""

    return {
        "query_id": f"query-{index:03d}",
        "user_id": user_id,
        "query_text": f"query {index}",
        "result_snapshot": {
            "query": f"query {index}",
            "events": ["person enters"],
            "results": [{
                "frame_id": FRAME_A,
                "video_id": "L21_V001",
                "frame_idx": 90,
                "timestamp_ms": 3_600,
                "score": 0.9,
                "frame_ids": [FRAME_A],
                "timestamps_ms": [3_600],
                "fps": 29.97,
                "rank": 1,
                "scores": {"visual": 0.9},
                "metadata": {
                    "title": "News",
                    "caption": "A person enters",
                    "ocr": "LIVE",
                    "objects": ["person"],
                    "asr": "Welcome",
                    "future_field": {"kept": True},
                },
            }],
            "latency": {"total_ms": 12.5},
            "warnings": ["example warning"],
        },
    }


def test_history_create_view_submit_and_replay(
    workspace_app: FastAPI,
    workspace_store: WorkspaceStore,
) -> None:
    """Persist a lossless snapshot and its deduplicated frame activity."""

    payload = history_payload(1)
    created = request(
        workspace_app,
        "POST",
        "/api/v1/query-history",
        json=payload,
    )
    assert created.status_code == 201
    assert created.json()["frame_activity"] == {
        "viewed_frame_ids": [],
        "submitted_frame_ids": [],
    }
    assert created.json()["result_snapshot"] == payload["result_snapshot"]

    for _ in range(2):
        viewed = request(
            workspace_app,
            "PATCH",
            "/api/v1/query-history/query-001/viewed-frame",
            json={"frame_id": FRAME_A},
        )
        assert viewed.status_code == 200
    assert viewed.json()["frame_activity"]["viewed_frame_ids"] == [FRAME_A]

    workspace_store.create_submission_file("query-01.csv", "L21_V001,90")
    submitted = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/query-001/submission",
        json={
            "submission_file_name": "query-01.csv",
            "submission_line": "L21_V001,90",
            "frame_ids": [FRAME_A, FRAME_B, FRAME_A],
        },
    )
    assert submitted.status_code == 200
    assert submitted.json()["submission_files"] == ["query-01.csv"]
    assert submitted.json()["frame_activity"]["submitted_frame_ids"] == [
        FRAME_A,
        FRAME_B,
    ]

    loaded = request(
        workspace_app,
        "GET",
        "/api/v1/query-history",
        params={"user_id": "user-a"},
    )
    assert loaded.json()["items"] == [submitted.json()]


def test_full_trake_snapshot_round_trips_without_frame_lookup_on_get(
    workspace_store: WorkspaceStore,
) -> None:
    """Preserve ordered TRAKE arrays and keep replay reads store-only."""

    service = FrameService()
    app = create_app(search_service=service, workspace_store=workspace_store)
    snapshot = {
        "events": ["person enters", "person sits"],
        "paths": [{
            "video_id": "L21_V001",
            "score": 0.8,
            "frame_ids": [FRAME_A, FRAME_B],
            "frame_idxs": [90, 91],
            "timestamps_ms": [3_600, 3_640],
            "future_path_field": {"kept": True},
        }],
        "latency": {"total_ms": 15.0},
    }
    created = request(app, "POST", "/api/v1/query-history", json={
        "query_id": "trake-001",
        "user_id": "user-a",
        "query_text": "ordered events",
        "result_snapshot": snapshot,
    })
    assert created.status_code == 201
    assert created.json()["result_snapshot"] == snapshot

    service.lookups.clear()
    loaded = request(
        app,
        "GET",
        "/api/v1/query-history",
        params={"user_id": "user-a"},
    )
    assert loaded.json()["items"][0]["result_snapshot"] == snapshot
    assert service.lookups == []


def test_history_validates_contract_and_canonical_frames(
    workspace_app: FastAPI,
) -> None:
    """Reject stale fields, missing frames, and mixed-video TRAKE paths."""

    stale = history_payload(1)
    stale["query_type"] = "kis"
    assert request(
        workspace_app, "POST", "/api/v1/query-history", json=stale
    ).status_code == 422

    invalid_snapshot = history_payload(3)
    invalid_snapshot["result_snapshot"] = {"events": []}
    assert request(
        workspace_app,
        "POST",
        "/api/v1/query-history",
        json=invalid_snapshot,
    ).status_code == 422

    missing = history_payload(2)
    missing["result_snapshot"]["results"][0]["frame_id"] = "missing"
    assert request(
        workspace_app, "POST", "/api/v1/query-history", json=missing
    ).status_code == 404

    trake = {
        "query_id": "trake-001",
        "user_id": "user-a",
        "query_text": "ordered events",
        "result_snapshot": {
            "paths": [{
                "video_id": "L99_V999",
                "score": 0.8,
                "frame_ids": [FRAME_A],
            }],
        },
    }
    assert request(
        workspace_app, "POST", "/api/v1/query-history", json=trake
    ).status_code == 422


def test_history_errors_and_latest_twenty(workspace_app: FastAPI) -> None:
    """Map conflicts and isolate each user's newest twenty histories."""

    payload = history_payload(0)
    assert request(
        workspace_app, "POST", "/api/v1/query-history", json=payload
    ).status_code == 201
    assert request(
        workspace_app, "POST", "/api/v1/query-history", json=payload
    ).status_code == 409

    for index in range(1, 22):
        assert request(
            workspace_app,
            "POST",
            "/api/v1/query-history",
            json=history_payload(index),
        ).status_code == 201
    assert request(
        workspace_app,
        "POST",
        "/api/v1/query-history",
        json=history_payload(99, user_id="user-b"),
    ).status_code == 201

    response = request(
        workspace_app,
        "GET",
        "/api/v1/query-history",
        params={"user_id": "user-a"},
    )
    assert len(response.json()["items"]) == 20
    assert response.json()["items"][0]["query_id"] == "query-021"

    other = request(
        workspace_app,
        "GET",
        "/api/v1/query-history",
        params={"user_id": "user-b"},
    )
    assert [item["query_id"] for item in other.json()["items"]] == ["query-099"]


def test_submission_requires_committed_file_line(
    workspace_app: FastAPI,
    workspace_store: WorkspaceStore,
) -> None:
    """Do not record a submission that is absent from shared file content."""

    request(
        workspace_app,
        "POST",
        "/api/v1/query-history",
        json=history_payload(1),
    )
    workspace_store.create_submission_file("query.csv", "L21_V001,120")

    response = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/query-001/submission",
        json={
            "submission_file_name": "query.csv",
            "submission_line": "L21_V001,90",
            "frame_ids": [FRAME_A],
        },
    )
    assert response.status_code == 409

    missing_history = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/missing/viewed-frame",
        json={"frame_id": FRAME_A},
    )
    missing_file = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/query-001/submission",
        json={
            "submission_file_name": "missing.csv",
            "submission_line": "L21_V001,90",
            "frame_ids": [FRAME_A],
        },
    )
    assert missing_history.status_code == 404
    assert missing_file.status_code == 404


def test_submission_file_state_revision_and_persistence(tmp_path: Path) -> None:
    """Persist shared files and protect edits with optimistic revisions."""

    database = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(database)
    empty = store.create_submission_file("query.csv", "")
    assert empty.revision == 1
    with pytest.raises(ValueError):
        store.validate_submission_file("query.csv", True, 1)

    filled = store.update_submission_file("query.csv", "L21_V001,90", 1)
    validated = store.validate_submission_file("query.csv", True, 2)
    assert filled.revision == 2
    assert validated.is_validated is True
    with pytest.raises(RevisionConflict) as conflict:
        store.update_submission_file("query.csv", "stale", 2)
    assert conflict.value.file.revision == 3

    reopened = WorkspaceStore(database)
    assert reopened.list_submission_files() == [validated]
    edited = reopened.update_submission_file("query.csv", "changed", 3)
    assert edited.is_validated is False


def test_submission_files_hydrate_and_websocket_broadcast(
    workspace_app: FastAPI,
) -> None:
    """Broadcast committed changes and return the latest conflict state."""

    with TestClient(workspace_app) as client:
        with client.websocket_connect("/api/v1/workspace/ws") as first:
            with client.websocket_connect("/api/v1/workspace/ws") as second:
                first.send_json({
                    "type": "submission_file.create",
                    "name": "shared.csv",
                    "content": "",
                })
                assert first.receive_json()["type"] == "submission_file.created"
                assert second.receive_json()["file"]["revision"] == 1

                second.send_json({
                    "type": "submission_file.update",
                    "name": "shared.csv",
                    "content": "L21_V001,90",
                    "expected_revision": 1,
                })
                assert first.receive_json()["file"]["revision"] == 2
                assert second.receive_json()["type"] == "submission_file.updated"

                first.send_json({
                    "type": "submission_file.update",
                    "name": "shared.csv",
                    "content": "stale",
                    "expected_revision": 1,
                })
                conflict = first.receive_json()
                assert conflict["type"] == "submission_file.conflict"
                assert conflict["file"]["content"] == "L21_V001,90"

        hydrated = client.get("/api/v1/submission-files")
        assert hydrated.json()["files"][0]["name"] == "shared.csv"


def test_workspace_websocket_rejects_unknown_origin(
    workspace_app: FastAPI,
) -> None:
    """Apply the configured browser-origin allowlist to WebSocket clients."""

    with TestClient(workspace_app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/api/v1/workspace/ws",
                headers={"origin": "https://untrusted.example"},
            ):
                pass
    assert rejected.value.code == 1008


def test_workspace_is_unavailable_without_store() -> None:
    """Keep unrelated APIs mountable when Workspace storage is not configured."""

    response = request(
        create_app(search_service=FrameService()),
        "GET",
        "/api/v1/submission-files",
    )
    assert response.status_code == 503


def test_history_store_reopens(tmp_path: Path) -> None:
    """Keep history after reopening the SQLite database."""

    database = tmp_path / "workspace.sqlite3"
    WorkspaceStore(database).create_history(
        QueryHistoryCreate.model_validate(history_payload(1))
    )
    reopened = WorkspaceStore(database)
    assert reopened.get_recent_history("user-a")[0].query_id == "query-001"
