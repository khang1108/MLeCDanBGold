"""Exact DRES v2 payload and durable-attempt route tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from hcmai.api.history import WorkspaceStore
from hcmai.api.routers.vbs import create_vbs_router
from hcmai.vbs.client import DresClient
from hcmai.vbs.config import DresCredential, DresSettings
from hcmai.vbs.service import DresService


def _app(
    tmp_path: Path,
    *,
    submit_status: int = 202,
    submit_statuses: list[int] | None = None,
    submit_verdict: str = "CORRECT",
):
    """Create a real DRES client/service over a deterministic mock server."""

    submissions: list[tuple[str, str, bytes]] = []
    login_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/api/v2/login":
            login_count += 1
            session_id = "private-session" if login_count == 1 else f"private-session-{login_count}"
            return httpx.Response(200, json={"sessionId": session_id})
        if request.url.path == "/api/v2/client/evaluation/currentTask/eval-1":
            return httpx.Response(200, json={
                "name": "KIS task",
                "taskGroup": "KIS",
                "taskType": "KIS",
            })
        if request.url.path == "/api/v2/submit/eval-1":
            submissions.append((request.url.path, request.url.params.get("session", ""), request.read()))
            request_status = (
                submit_statuses[len(submissions) - 1]
                if submit_statuses is not None and len(submissions) <= len(submit_statuses)
                else submit_status
            )
            if request_status in {200, 202}:
                return httpx.Response(request_status, json={
                    "status": True,
                    "submission": submit_verdict,
                    "description": "accepted",
                })
            return httpx.Response(
                request_status,
                json={"status": False, "description": "safe DRES response"},
            )
        raise AssertionError(f"Unexpected DRES request {request.method} {request.url.path}")

    settings = DresSettings(
        base_url="https://dres.test",
        evaluation_id="eval-1",
        credentials={
            "member-1": DresCredential(
                username="backend-only-user",
                password=SecretStr("backend-only-password"),
            ),
        },
        media_id_prefix_to_strip="prefix.",
    )
    dres_client = DresClient(settings, transport=httpx.MockTransport(handler))
    dres_service = DresService(settings, client=dres_client)
    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    app = FastAPI()
    app.include_router(create_vbs_router({
        "vbs_service": dres_service,
        "workspace_store": store,
    }))
    return app, store, dres_service, submissions


def _scope_key() -> str:
    """Return the deterministic internal key for this route fixture's task."""

    canonical = json.dumps(
        {
            "evaluation_id": "eval-1",
            "name": "KIS task",
            "taskGroup": "KIS",
            "taskType": "KIS",
            "duration": None,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"dres-task-v1:{hashlib.sha256(canonical).hexdigest()}"


def _connect(client: TestClient) -> None:
    """Perform the browser-safe user-ID-only handshake."""

    response = client.post(
        "/api/v1/vbs/session/connect",
        json={"user_id": "member-1"},
    )
    assert response.status_code == 200
    assert response.json() == {"user_id": "member-1", "connected": True}
    assert "private-session" not in response.text
    assert "backend-only" not in response.text


def test_connect_reports_missing_dres_configuration_without_exposing_secrets() -> None:
    """Keep the route mounted and return a safe config error without DRES env."""

    app = FastAPI()
    app.include_router(create_vbs_router({"vbs_service": None}))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/vbs/session/connect",
            json={"user_id": "member-1"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "DRES session service is not configured"}


@pytest.mark.parametrize(
    "submit_status,verdict",
    [
        (200, "CORRECT"),
        (202, "WRONG"),
        (200, "INDETERMINATE"),
        (202, "UNDECIDABLE"),
    ],
)
def test_kis_submits_exact_point_and_preserves_each_dres_verdict(
    tmp_path: Path,
    submit_status: int,
    verdict: str,
) -> None:
    """Ignore frame IDs and other coordinates; preserve exact milliseconds."""

    app, store, service, requests = _app(
        tmp_path,
        submit_status=submit_status,
        submit_verdict=verdict,
    )
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_frame_candidate(
            video_id="prefix.video.with.dots",
            timestamp_ms=12_346,
            source_frame_id="frame-9001",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/kis", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "ACCEPTED"
    assert response.json()["accepted"] is True
    assert response.json()["verdict"] == verdict
    assert requests == [(
        "/api/v2/submit/eval-1",
        "private-session",
        b'{"answerSets":[{"taskName":"KIS task","answers":[{"mediaItemName":"video.with.dots","start":12346,"end":12346}]}]}',
    )]
    assert "private-session" not in response.text
    assert "backend-only" not in response.text
    workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
    assert workspace.candidates[0].submitted_at_ms is not None
    asyncio.run(service.aclose())


def test_vqa_submits_text_only_in_one_answer_set(tmp_path: Path) -> None:
    """Keep VQA answer text free of temporal media coordinates."""

    app, store, service, requests = _app(tmp_path)
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_text_candidate(
            text="the red vehicle turns left",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/vqa", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 200, response.text
    assert requests == [(
        "/api/v2/submit/eval-1",
        "private-session",
        b'{"answerSets":[{"taskName":"KIS task","answers":[{"text":"the red vehicle turns left"}]}]}',
    )]
    asyncio.run(service.aclose())


def test_avs_posts_all_ordered_frame_answers_in_one_request(tmp_path: Path) -> None:
    """Send one exact answerSet containing the full ordered AVS workspace."""

    app, store, service, requests = _app(tmp_path)
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        mode = store.set_avs_enabled(
            avs_enabled=True,
            expected_workspace_revision=workspace.revision,
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
        )
        first = store.add_frame_candidate(
            video_id="video-a",
            timestamp_ms=98_765,
            source_frame_id="frame-9001",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=mode.workspace_revision,
        ).candidate
        time.sleep(0.003)
        second = store.add_frame_candidate(
            video_id="prefix.video-b",
            timestamp_ms=1_234,
            source_frame_id="frame-2",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=mode.workspace_revision + 1,
        ).candidate
        current = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")

        response = client.post("/api/v1/vbs/submit/avs", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": current.revision,
            "candidates": [
                {"candidate_id": first.candidate_id, "expected_revision": first.revision},
                {"candidate_id": second.candidate_id, "expected_revision": second.revision},
            ],
        })

    assert response.status_code == 200, response.text
    assert len(requests) == 1
    assert requests[0] == (
        "/api/v2/submit/eval-1",
        "private-session",
        b'{"answerSets":[{"taskName":"KIS task","answers":[{"mediaItemName":"video-a","start":98765,"end":98765},{"mediaItemName":"video-b","start":1234,"end":1234}]}]}',
    )
    asyncio.run(service.aclose())


@pytest.mark.parametrize("status_code", [400, 401, 404, 412])
def test_definitive_dres_rejection_clears_reservation_without_marking_answer(
    tmp_path: Path,
    status_code: int,
) -> None:
    """Keep the workspace editable after a definitive upstream rejection."""

    app, store, service, requests = _app(tmp_path, submit_status=status_code)
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_frame_candidate(
            video_id="video-a",
            timestamp_ms=9_999,
            source_frame_id=None,
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/kis", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 502
    latest = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
    assert latest.pending_submission is None
    assert latest.candidates[0].submitted_at_ms is None
    assert len(requests) == (2 if status_code == 401 else 1)
    asyncio.run(service.aclose())


def test_ambiguous_dres_failure_keeps_unknown_reservation_locked(tmp_path: Path) -> None:
    """Do not retry or release a request whose network outcome is ambiguous."""

    app, store, service, requests = _app(tmp_path, submit_status=503)
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_frame_candidate(
            video_id="video-a",
            timestamp_ms=7_654,
            source_frame_id=None,
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/kis", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 504
    assert response.json()["state"] == "UNKNOWN"
    latest = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
    assert latest.pending_submission is not None
    assert latest.pending_submission.state == "UNKNOWN"
    assert len(requests) == 1
    asyncio.run(service.aclose())


def test_submit_401_relogs_once_then_accepts_the_same_frozen_snapshot(tmp_path: Path) -> None:
    """Retry the exact reserved request once after DRES definitively expires its session."""

    app, store, service, requests = _app(tmp_path, submit_statuses=[401, 202])
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_frame_candidate(
            video_id="video-a",
            timestamp_ms=4_321,
            source_frame_id=None,
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/kis", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "ACCEPTED"
    assert [request[1] for request in requests] == ["private-session", "private-session-2"]
    assert requests[0][2] == requests[1][2]
    workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
    assert workspace.pending_submission is None
    assert workspace.candidates[0].submitted_at_ms is not None
    asyncio.run(service.aclose())


@pytest.mark.parametrize(
    "changed_scope",
    [
        ("eval-1", "dres-task-v1:changed"),
        ("eval-after-reservation", _scope_key()),
    ],
)
def test_scope_change_after_reservation_releases_without_posting_to_dres(
    tmp_path: Path,
    changed_scope: tuple[str, str],
) -> None:
    """Release the unsent snapshot if its active evaluation or task changes."""

    app, store, service, requests = _app(tmp_path)
    original_resolve_scope = service.resolve_scope
    resolve_count = 0

    async def changed_scope_on_final_check(user_id: str):
        nonlocal resolve_count
        resolved = await original_resolve_scope(user_id)
        resolve_count += 1
        if resolve_count == 3:
            return resolved.model_copy(update={
                "evaluation_id": changed_scope[0],
                "task_scope_key": changed_scope[1],
            })
        return resolved

    service.resolve_scope = changed_scope_on_final_check
    with TestClient(app) as client:
        _connect(client)
        workspace = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
        candidate = store.add_frame_candidate(
            video_id="video-a",
            timestamp_ms=8_765,
            source_frame_id="frame-1",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key=_scope_key(),
            task_name="KIS task",
            expected_workspace_revision=workspace.revision,
        ).candidate

        response = client.post("/api/v1/vbs/submit/kis", json={
            "user_id": "member-1",
            "task_scope_key": _scope_key(),
            "expected_workspace_revision": candidate.revision + 1,
            "candidate_id": candidate.candidate_id,
            "expected_revision": candidate.revision,
        })

    assert response.status_code == 409
    assert response.json()["state"] == "NOT_ACCEPTED"
    assert response.json()["accepted"] is False
    assert resolve_count == 3
    assert requests == []
    latest = store.get_answer_workspace("eval-1", _scope_key(), "KIS task")
    assert latest.pending_submission is None
    assert latest.candidates[0].submitted_at_ms is None
    attempt = store.get_submission_attempt(response.json()["attempt_id"])
    assert attempt.state == "NOT_ACCEPTED"
    asyncio.run(service.aclose())
