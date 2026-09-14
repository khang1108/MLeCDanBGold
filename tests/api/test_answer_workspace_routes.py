"""HTTP and WebSocket tests for the structured answer workspace transport."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from hcmai.api.contracts.workspace import AnswerWorkspaceEvent
from hcmai.api.history import WorkspaceStore
from hcmai.api.routers.workspace import create_answer_workspace_router
from hcmai.corpus.models import Frame
from hcmai.vbs.models import DresTaskScope


class _VbsService:
    """Provide a stable active task and browser-safe connection state."""

    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.scopes: dict[str, tuple[str, str]] = {}

    def session_status(self, user_id: str) -> dict[str, bool | str]:
        return {"user_id": user_id, "connected": self.connected}

    async def resolve_scope(self, user_id: str) -> DresTaskScope:
        evaluation_id, scope_key = self.scopes.get(user_id, ("eval-1", "task-1"))
        return DresTaskScope(
            evaluation_id=evaluation_id,
            task_scope_key=scope_key,
            task_name=f"Task {scope_key}",
            task_group="KIS",
            task_type="KIS",
        )


class _SearchService:
    """Expose one frame for provenance validation in card-origin mutations."""

    def get_frame(self, frame_id: str) -> Frame:
        if frame_id != "frame-1":
            raise KeyError(frame_id)
        return Frame(
            frame_id="frame-1",
            video_id="video-1",
            frame_idx=77,
            timestamp_ms=12000,
            image_path="/frame-1.jpg",
        )


def _app(tmp_path: Path, *, connected: bool = True) -> FastAPI:
    """Create an isolated router app without loading retrieval artifacts."""

    app = FastAPI()
    app.include_router(create_answer_workspace_router({
        "workspace_store": WorkspaceStore(tmp_path / "workspace.sqlite3"),
        "vbs_service": _VbsService(connected),
        "service": _SearchService(),
    }))
    return app


def _workspace_db_path() -> Path:
    """Choose a unique test database in the workspace root for restricted runners."""

    return Path.cwd() / f".answer-workspace-test-{uuid.uuid4().hex}.sqlite3"


def _remove_workspace_db(database_path: Path) -> None:
    """Remove one test's SQLite file and its optional WAL companions."""

    for suffix in ("", "-wal", "-shm"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)


def test_http_hydration_requires_connected_user_and_returns_only_workspace(tmp_path: Path) -> None:
    """Hydrate from SQLite for the connected participant without DRES secrets."""

    app = _app(tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/answer-workspace",
            headers={"X-VBS-User-ID": "member-1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["evaluation_id"] == "eval-1"
    assert body["task_scope_key"] == "task-1"
    assert body["candidates"] == []
    assert "session" not in response.text.lower()


def test_http_hydration_rejects_disconnected_user(tmp_path: Path) -> None:
    """Do not expose shared mutations to a browser without a DRES handshake."""

    app = _app(tmp_path, connected=False)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/answer-workspace",
            headers={"X-VBS-User-ID": "member-1"},
        )

    assert response.status_code == 401


def test_websocket_accepts_public_user_id_query_for_native_browser_clients(
) -> None:
    """Native browser WebSocket clients can identify a participant without custom headers."""

    database_path = _workspace_db_path()
    try:
        services = {
            "workspace_store": WorkspaceStore(database_path),
            "vbs_service": _VbsService(),
            "service": _SearchService(),
        }
        app = FastAPI()
        app.include_router(create_answer_workspace_router(services))
        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/answer-workspace/ws?user_id=member-1",
            ) as socket:
                event = socket.receive_json()
    finally:
        _remove_workspace_db(database_path)

    assert event["type"] == "answer.workspace.hydrated"
    assert event["workspace"]["active_task_scope_key"] == "task-1"


def test_websocket_hydrates_then_broadcasts_a_revision_checked_exact_frame(
    tmp_path: Path,
) -> None:
    """Preserve exact inspector milliseconds and broadcast only committed state."""

    app = _app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/answer-workspace/ws",
            headers={"X-VBS-User-ID": "member-1"},
        ) as first:
            initial = first.receive_json()
            assert initial["type"] == "answer.workspace.hydrated"
            revision = initial["workspace"]["revision"]
            with client.websocket_connect(
                "/api/v1/answer-workspace/ws",
                headers={"X-VBS-User-ID": "member-2"},
            ) as second:
                assert second.receive_json()["type"] == "answer.workspace.hydrated"
                first.send_json({
                    "type": "answer.add_frame",
                    "expected_workspace_revision": revision,
                    "video_id": "video-live",
                    "timestamp_ms": 12346,
                    "source_frame_id": None,
                })
                event = first.receive_json()
                mirrored = second.receive_json()

    assert event == mirrored
    assert event["type"] == "answer.added"
    assert event["workspace"]["candidates"][0]["video_id"] == "video-live"
    assert event["workspace"]["candidates"][0]["timestamp_ms"] == 12346
    assert event["workspace"]["candidates"][0]["source_frame_id"] is None


def test_websocket_replays_commit_that_lands_during_initial_hydration(
    monkeypatch,
) -> None:
    """A commit between snapshot read and send must follow hydration, not precede it."""

    database_path = _workspace_db_path()
    try:
        store = WorkspaceStore(database_path)
        services = {
            "workspace_store": store,
            "vbs_service": _VbsService(),
            "service": _SearchService(),
        }
        app = FastAPI()
        app.include_router(create_answer_workspace_router(services))
        original_send_json = WebSocket.send_json
        injected = False

        async def commit_while_hydration_is_being_sent(self, data, mode="text") -> None:
            nonlocal injected
            if not injected and isinstance(data, dict) and data.get("type") == "answer.workspace.hydrated":
                injected = True
                before = store.get_answer_workspace("eval-1", "task-1", "Task task-1")
                store.add_frame_candidate(
                    video_id="video-during-hydration",
                    timestamp_ms=2345,
                    source_frame_id=None,
                    user_id="other-member",
                    evaluation_id="eval-1",
                    task_scope_key="task-1",
                    task_name="Task task-1",
                    expected_workspace_revision=before.revision,
                )
                committed = store.get_answer_workspace("eval-1", "task-1", "Task task-1")
                await services["broadcast_workspace"](AnswerWorkspaceEvent(
                    type="answer.added",
                    workspace=committed,
                ))
            await original_send_json(self, data, mode=mode)

        monkeypatch.setattr(WebSocket, "send_json", commit_while_hydration_is_being_sent)

        with TestClient(app) as client:
            with client.websocket_connect(
                "/api/v1/answer-workspace/ws",
                headers={"X-VBS-User-ID": "member-1"},
            ) as socket:
                hydrated = socket.receive_json()
                committed = socket.receive_json()

        assert injected
        assert hydrated["type"] == "answer.workspace.hydrated"
        assert committed["type"] == "answer.added"
        assert committed["workspace"]["revision"] > hydrated["workspace"]["revision"]
        assert committed["workspace"]["candidates"][0]["video_id"] == "video-during-hydration"
    finally:
        _remove_workspace_db(database_path)


def test_connected_old_scope_collaborators_receive_clear_and_switch_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Rebind old-task sockets after an explicit committed workspace switch."""

    service = _VbsService()
    old_scope = ("eval-old", "task-old")
    new_scope = ("eval-new", "task-new")
    unrelated_scope = ("eval-other", "task-other")
    service.scopes.update({
        "initiator": old_scope,
        "collaborator": old_scope,
        "new-member": new_scope,
        "unrelated": unrelated_scope,
    })
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    initial = store.get_answer_workspace(old_scope[0], old_scope[1], f"Task {old_scope[1]}")
    store.add_text_candidate(
        text="legacy task answer",
        user_id="initiator",
        evaluation_id=old_scope[0],
        task_scope_key=old_scope[1],
        task_name=f"Task {old_scope[1]}",
        expected_workspace_revision=initial.revision,
    )
    before_switch = store.get_answer_workspace(old_scope[0], old_scope[1], f"Task {old_scope[1]}")
    services = {"workspace_store": store, "vbs_service": service}
    app = FastAPI()
    app.include_router(create_answer_workspace_router(services))
    switched_recipients: list[str | None] = []
    original_send_json = WebSocket.send_json

    async def record_task_switch(self, data, mode="text") -> None:
        if isinstance(data, dict) and data.get("type") == "answer.task.switched":
            switched_recipients.append(self.headers.get("x-vbs-user-id"))
        await original_send_json(self, data, mode=mode)

    monkeypatch.setattr(WebSocket, "send_json", record_task_switch)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/answer-workspace/ws",
            headers={"X-VBS-User-ID": "initiator"},
        ) as initiator:
            assert initiator.receive_json()["workspace"]["task_scope_key"] == old_scope[1]
            with client.websocket_connect(
                "/api/v1/answer-workspace/ws",
                headers={"X-VBS-User-ID": "collaborator"},
            ) as collaborator:
                assert collaborator.receive_json()["workspace"]["task_scope_key"] == old_scope[1]
                with client.websocket_connect(
                    "/api/v1/answer-workspace/ws",
                    headers={"X-VBS-User-ID": "new-member"},
                ) as new_member:
                    assert new_member.receive_json()["workspace"]["active_task_scope_key"] == new_scope[1]
                    with client.websocket_connect(
                        "/api/v1/answer-workspace/ws",
                        headers={"X-VBS-User-ID": "unrelated"},
                    ) as unrelated:
                        assert unrelated.receive_json()["workspace"]["active_task_scope_key"] == unrelated_scope[1]

                        # DRES has advanced while the first two sockets still
                        # carry their old registration scope.
                        service.scopes["initiator"] = new_scope
                        service.scopes["collaborator"] = new_scope
                        initiator.send_json({
                            "type": "answer.task.clear_and_switch",
                            "expected_workspace_revision": before_switch.revision,
                            "expected_old_evaluation_id": old_scope[0],
                            "expected_old_task_scope_key": old_scope[1],
                            "target_evaluation_id": new_scope[0],
                            "target_task_scope_key": new_scope[1],
                            "target_task_name": f"Task {new_scope[1]}",
                        })

                        # Wait for the durable commit before checking socket
                        # membership, so the assertion fails promptly on the
                        # old hub behavior instead of blocking on receive_json.
                        deadline = time.monotonic() + 2
                        hub = services["workspace_socket_hub"]
                        scopes = list(hub._sockets.values())
                        while scopes.count(new_scope) != 3 and time.monotonic() < deadline:
                            time.sleep(0.01)
                            scopes = list(hub._sockets.values())

                        assert scopes.count(new_scope) == 3
                        assert scopes.count(unrelated_scope) == 1
                        assert old_scope not in scopes

                        initiator_event = initiator.receive_json()
                        collaborator_event = collaborator.receive_json()
                        new_member_event = new_member.receive_json()

    assert initiator_event == collaborator_event == new_member_event
    assert initiator_event["type"] == "answer.task.switched"
    assert initiator_event["workspace"]["task_scope_key"] == new_scope[1]
    assert initiator_event["workspace"]["candidates"] == []
    assert switched_recipients == ["initiator", "collaborator", "new-member"]
