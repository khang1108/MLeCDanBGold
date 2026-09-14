"""Integration tests for SQLite query replay and viewed-frame activity."""

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
from hcmai.api.history import WorkspaceStore
from hcmai.app import create_app
from hcmai.corpus.models import Frame


FRAME_A = "L21_V001_00000090"
FRAME_B = "L21_V001_00000120"


class FrameService:
    """Expose canonical frame lookup without loading search artifacts."""

    llm = None
    reranking = None

    def __init__(self) -> None:
        """Build two hand-checkable canonical frame rows."""

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
        """Return one known canonical frame or report its missing identity."""

        self.lookups.append(frame_id)
        if frame_id not in self.frames:
            raise KeyError(frame_id)
        return self.frames[frame_id]

    def health(self, messages: list[str]) -> dict[str, Any]:
        """Return fields consumed by app startup logging."""

        del messages
        return {"capabilities": {"search": False}, "remote_inference": {}}

    def close(self) -> None:
        """Provide the application lifecycle hook."""


class VbsSessionService:
    """Supply one connected participant scope to workspace WebSocket tests."""

    def session_status(self, user_id: str) -> dict[str, bool | str]:
        """Return a browser-safe connected state."""

        return {"user_id": user_id, "connected": True}

    async def resolve_scope(self, user_id: str) -> tuple[str, str, object]:
        """Return a stable live DRES scope for the workspace test."""

        del user_id
        return "eval-1", "task-1", object()


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
    """Create an isolated SQLite store."""

    return WorkspaceStore(tmp_path / "workspace.sqlite3")


@pytest.fixture
def workspace_app(workspace_store: WorkspaceStore) -> FastAPI:
    """Create an app with canonical frame lookup and workspace storage."""

    return create_app(
        search_service=FrameService(),
        workspace_store=workspace_store,
        vbs_service=VbsSessionService(),
    )


def request(app: FastAPI, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Send one request through ASGI without starting online services."""

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


def test_history_create_view_and_replay(
    workspace_app: FastAPI,
) -> None:
    """Persist a lossless snapshot and deduplicated viewed-frame activity."""

    payload = history_payload(1)
    created = request(
        workspace_app,
        "POST",
        "/api/v1/query-history",
        json=payload,
    )
    assert created.status_code == 201
    assert created.json()["frame_activity"] == {"viewed_frame_ids": []}
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

    loaded = request(
        workspace_app,
        "GET",
        "/api/v1/query-history",
        params={"user_id": "user-a"},
    )
    assert loaded.json()["items"] == [viewed.json()]


def test_full_trake_snapshot_round_trips_without_frame_lookup_on_get(
    workspace_store: WorkspaceStore,
) -> None:
    """Preserve ordered legacy TRAKE arrays for stored history replay."""

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


def test_history_validates_contract_and_canonical_frames(workspace_app: FastAPI) -> None:
    """Reject stale fields, missing canonical rows, and mismatched TRAKE paths."""

    stale = history_payload(1)
    stale["query_type"] = "kis"
    assert request(workspace_app, "POST", "/api/v1/query-history", json=stale).status_code == 422

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
    assert request(workspace_app, "POST", "/api/v1/query-history", json=missing).status_code == 404

    trake = {
        "query_id": "trake-001",
        "user_id": "user-a",
        "query_text": "ordered events",
        "result_snapshot": {
            "paths": [{"video_id": "L99_V999", "score": 0.8, "frame_ids": [FRAME_A]}],
        },
    }
    assert request(workspace_app, "POST", "/api/v1/query-history", json=trake).status_code == 422


def test_history_errors_and_latest_twenty(workspace_app: FastAPI) -> None:
    """Reject duplicate query IDs and isolate each participant's recent searches."""

    payload = history_payload(0)
    assert request(workspace_app, "POST", "/api/v1/query-history", json=payload).status_code == 201
    assert request(workspace_app, "POST", "/api/v1/query-history", json=payload).status_code == 409

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


def test_missing_query_history_and_viewed_frame_return_not_found(
    workspace_app: FastAPI,
) -> None:
    """Return a stable not-found response for missing history or frame IDs."""

    response = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/missing/viewed-frame",
        json={"frame_id": FRAME_A},
    )
    assert response.status_code == 404
    missing_frame = request(
        workspace_app,
        "PATCH",
        "/api/v1/query-history/missing/viewed-frame",
        json={"frame_id": "missing-frame"},
    )
    assert missing_frame.status_code == 404


def test_workspace_websocket_rejects_unknown_origin(workspace_app: FastAPI) -> None:
    """Apply the configured browser-origin allowlist to answer workspace sockets."""

    with TestClient(workspace_app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/api/v1/answer-workspace/ws",
                headers={
                    "origin": "https://untrusted.example",
                    "X-VBS-User-ID": "member-1",
                },
            ):
                pass
    assert rejected.value.code == 1008


def test_answer_workspace_requires_storage_but_other_apis_remain_mounted() -> None:
    """Keep retrieval APIs mounted while an unconfigured workspace returns 503."""

    response = request(
        create_app(search_service=FrameService(), vbs_service=VbsSessionService()),
        "GET",
        "/api/v1/answer-workspace",
        headers={"X-VBS-User-ID": "member-1"},
    )
    assert response.status_code == 503


def test_history_store_reopens(tmp_path: Path) -> None:
    """Keep query replay data after reopening the SQLite database."""

    database = tmp_path / "workspace.sqlite3"
    WorkspaceStore(database).create_history(
        QueryHistoryCreate.model_validate(history_payload(1))
    )
    reopened = WorkspaceStore(database)
    assert reopened.get_recent_history("user-a")[0].query_id == "query-001"
