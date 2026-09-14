"""Contract and route tests for stateless private-session DRES submission."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from hcmai.api.contracts.vbs import (
    VbsDirectSubmissionNotRecorded,
    VbsDirectSubmissionRecorded,
    VbsDirectSubmissionRequest,
    VbsDirectSubmissionUnknown,
    VbsTaskResponse,
)
from hcmai.api.routers.vbs import create_vbs_router
from hcmai.vbs.client import DresClient
from hcmai.vbs.config import DresCredential, DresSettings
from hcmai.vbs.service import DresService


def _harness(
    *,
    task_type: str = "KIS",
    task_group: str = "KIS",
    submit_statuses: list[int] | None = None,
    submit_verdict: str = "CORRECT",
    malformed_success: bool = False,
):
    """Build the real route/service/client chain over a controllable DRES fake."""

    state: dict[str, Any] = {
        "task": {
            "name": f"{task_type} task",
            "taskGroup": task_group,
            "taskType": task_type,
            "duration": 300,
        },
        "submissions": [],
        "task_reads": [],
        "logins": Counter(),
        "submit_statuses": list(submit_statuses or [202]),
        "submit_verdict": submit_verdict,
        "malformed_success": malformed_success,
    }
    user_sessions = {
        "member-1": "private-session-member-1",
        "member-2": "private-session-member-2",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/login":
            credentials = json.loads(request.read())
            username = credentials["username"]
            state["logins"][username] += 1
            user_id = username.removeprefix("dres-")
            return httpx.Response(200, json={"sessionId": user_sessions[user_id]})

        if request.url.path == "/api/v2/client/evaluation/currentTask/eval-1":
            state["task_reads"].append(request.url.params.get("session", ""))
            return httpx.Response(200, json=dict(state["task"]))

        if request.url.path == "/api/v2/submit/eval-1":
            state["submissions"].append((
                request.url.params.get("session", ""),
                request.read(),
            ))
            status_code = (
                state["submit_statuses"].pop(0)
                if state["submit_statuses"]
                else 202
            )
            if status_code in {200, 202}:
                if state["malformed_success"]:
                    return httpx.Response(status_code, json={"status": True})
                return httpx.Response(status_code, json={
                    "status": True,
                    "submission": state["submit_verdict"],
                    "description": "accepted",
                })
            return httpx.Response(
                status_code,
                json={"status": False, "description": "mock rejection"},
            )

        raise AssertionError(f"Unexpected DRES request {request.method} {request.url.path}")

    settings = DresSettings(
        base_url="https://dres.test",
        evaluation_id="eval-1",
        credentials={
            "member-1": DresCredential(
                username="dres-member-1",
                password=SecretStr("backend-password-1"),
            ),
            "member-2": DresCredential(
                username="dres-member-2",
                password=SecretStr("backend-password-2"),
            ),
        },
        media_id_prefix_to_strip="prefix.",
    )
    service = DresService(
        settings,
        client=DresClient(settings, transport=httpx.MockTransport(handler)),
    )
    app = FastAPI()
    app.include_router(create_vbs_router({"vbs_service": service}))
    return app, service, state


@pytest.fixture
def api():
    """Provide an isolated browser-to-DRES test app and close its HTTP client."""

    app, service, state = _harness()
    with TestClient(app) as client:
        yield client, service, state
    asyncio.run(service.aclose())


def _connect(client: TestClient, user_id: str = "member-1") -> None:
    """Connect one configured participant without exposing its private token."""

    response = client.post(
        "/api/v1/vbs/session/connect",
        json={"user_id": user_id},
    )
    assert response.status_code == 200
    assert response.json() == {"user_id": user_id, "connected": True}
    assert "private-session" not in response.text
    assert "backend-password" not in response.text


def _current_task(client: TestClient, user_id: str = "member-1") -> dict[str, Any]:
    """Fetch the safe live task contract for one connected participant."""

    response = client.get(f"/api/v1/vbs/task/{user_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _temporal_request(
    scope: dict[str, Any],
    *,
    user_id: str = "member-1",
    video_id: str = "prefix.L21_V001",
    start_ms: int = 1_234,
    end_ms: int = 5_678,
) -> dict[str, Any]:
    """Build one participant-scoped temporal request from a task response."""

    return {
        "user_id": user_id,
        "expected_task_scope_key": scope["task_scope_key"],
        "answer": {
            "kind": "TEMPORAL",
            "video_id": video_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "user_id": "member-1",
            "expected_task_scope_key": "dres-task-v1:opaque",
            "answer": {
                "kind": "TEMPORAL",
                "video_id": "L21_V001",
                "start_ms": 1_234,
                "end_ms": 1_234,
            },
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "dres-task-v1:opaque",
            "answer": {"kind": "TEXT", "text": "three"},
        },
    ],
)
def test_direct_submission_contract_accepts_exactly_one_discriminated_answer(
    payload: dict[str, Any],
) -> None:
    """Accept either one temporal answer or one text answer, never a batch."""

    request = VbsDirectSubmissionRequest.model_validate(payload)

    assert request.answer.kind in {"TEMPORAL", "TEXT"}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "user_id": " ",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEXT", "text": "three"},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": " ",
            "answer": {"kind": "TEXT", "text": "three"},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEXT", "text": "  "},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEMPORAL", "video_id": " ", "start_ms": 0, "end_ms": 0},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEMPORAL", "video_id": "video", "start_ms": -1, "end_ms": 0},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEMPORAL", "video_id": "video", "start_ms": 0, "end_ms": 1.5},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEMPORAL", "video_id": "video", "start_ms": 3, "end_ms": 2},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEXT", "text": "three", "video_id": "video"},
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answers": [{"kind": "TEXT", "text": "three"}],
        },
        {
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "OTHER", "value": "three"},
        },
    ],
)
def test_direct_submission_contract_rejects_invalid_or_plural_answers(
    payload: dict[str, Any],
) -> None:
    """Reject blank, malformed, extra, missing, and plural answer fields."""

    with pytest.raises(ValidationError):
        VbsDirectSubmissionRequest.model_validate(payload)


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            VbsDirectSubmissionRecorded,
            {"state": "RECORDED", "recorded": True, "verdict": "CORRECT", "message": "ok"},
        ),
        (
            VbsDirectSubmissionNotRecorded,
            {
                "state": "NOT_RECORDED",
                "recorded": False,
                "verdict": None,
                "reason": "DRES_REJECTED",
                "message": "rejected",
            },
        ),
        (
            VbsDirectSubmissionNotRecorded,
            {
                "state": "NOT_RECORDED",
                "recorded": False,
                "verdict": None,
                "reason": "DRES_AUTH_REJECTED",
                "message": "reconnect",
            },
        ),
        (
            VbsDirectSubmissionUnknown,
            {"state": "UNKNOWN", "recorded": None, "verdict": None, "message": "check DRES"},
        ),
    ],
)
def test_direct_submission_outcome_contract_accepts_consistent_states(
    model: type,
    payload: dict[str, Any],
) -> None:
    """Represent definitive and ambiguous delivery without contradictory flags."""

    assert model.model_validate(payload)


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            VbsDirectSubmissionRecorded,
            {"state": "RECORDED", "recorded": False, "verdict": "CORRECT", "message": "bad"},
        ),
        (
            VbsDirectSubmissionRecorded,
            {"state": "RECORDED", "recorded": True, "message": "bad"},
        ),
        (
            VbsDirectSubmissionNotRecorded,
            {
                "state": "NOT_RECORDED",
                "recorded": True,
                "verdict": "WRONG",
                "reason": "DRES_REJECTED",
                "message": "bad",
            },
        ),
        (
            VbsDirectSubmissionNotRecorded,
            {"state": "NOT_RECORDED", "recorded": False, "verdict": None, "message": "bad"},
        ),
        (
            VbsDirectSubmissionNotRecorded,
            {
                "state": "NOT_RECORDED",
                "recorded": False,
                "reason": "DRES_REJECTED",
                "message": "bad",
            },
        ),
        (
            VbsDirectSubmissionUnknown,
            {"state": "UNKNOWN", "recorded": False, "verdict": None, "message": "bad"},
        ),
        (
            VbsDirectSubmissionUnknown,
            {"state": "UNKNOWN", "recorded": None, "verdict": "CORRECT", "message": "bad"},
        ),
    ],
)
def test_direct_submission_outcome_contract_rejects_contradictory_states(
    model: type,
    payload: dict[str, Any],
) -> None:
    """Do not let outcome validation blur recorded, rejected, and unknown."""

    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_current_task_route_returns_safe_scope_without_credentials_or_session(api) -> None:
    """Expose opaque task scope and competition metadata after a participant connects."""

    client, _service, _state = api
    _connect(client)

    response = client.get("/api/v1/vbs/task/member-1")

    assert response.status_code == 200, response.text
    task = VbsTaskResponse.model_validate(response.json())
    assert task.user_id == "member-1"
    assert task.evaluation_id == "eval-1"
    assert task.task_scope_key.startswith("dres-task-v1:")
    assert task.task_name == "KIS task"
    assert task.task_group == "KIS"
    assert task.task_type == "KIS"
    assert task.duration == 300
    assert "private-session" not in response.text
    assert "backend-password" not in response.text
    assert "dres-member" not in response.text


def test_current_task_route_requires_a_connected_participant(api) -> None:
    """Use a stable structured error instead of attempting implicit login."""

    client, _service, state = api

    response = client.get("/api/v1/vbs/task/member-1")

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "PARTICIPANT_NOT_CONNECTED"
    assert state["task_reads"] == []


@pytest.mark.parametrize("task_type", ["KIS", "AVS"])
def test_kis_and_avs_send_one_exact_temporal_answer(
    task_type: str,
) -> None:
    """Map a canonical frame to one ranged DRES temporal answer for each task."""

    app, service, state = _harness(task_type=task_type, task_group=task_type)
    try:
        with TestClient(app) as client:
            _connect(client)
            scope = _current_task(client)

            response = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(scope),
            )

        assert response.status_code == 200, response.text
        assert response.json()["state"] == "RECORDED"
        assert response.json()["recorded"] is True
        assert response.json()["verdict"] == "CORRECT"
        assert len(state["submissions"]) == 1
        session, body = state["submissions"][0]
        assert session == "private-session-member-1"
        assert json.loads(body) == {
            "answerSets": [{
                "taskName": f"{task_type} task",
                "answers": [{
                    "mediaItemName": "L21_V001",
                    "start": 1_234,
                    "end": 5_678,
                }],
            }],
        }
        assert "private-session" not in response.text
        assert "backend-password" not in response.text
    finally:
        asyncio.run(service.aclose())


def test_vqa_sends_one_text_answer_and_no_temporal_coordinates() -> None:
    """Keep VQA's single text answer separate from temporal media identity."""

    app, service, state = _harness(task_type="VQA", task_group="VQA")
    try:
        with TestClient(app) as client:
            _connect(client)
            scope = _current_task(client)
            response = client.post("/api/v1/vbs/submit", json={
                "user_id": "member-1",
                "expected_task_scope_key": scope["task_scope_key"],
                "answer": {"kind": "TEXT", "text": "three"},
            })

        assert response.status_code == 200, response.text
        assert response.json()["state"] == "RECORDED"
        assert len(state["submissions"]) == 1
        assert json.loads(state["submissions"][0][1]) == {
            "answerSets": [{"taskName": "VQA task", "answers": [{"text": "three"}]}],
        }
    finally:
        asyncio.run(service.aclose())


@pytest.mark.parametrize(
    "task_type,answer",
    [
        ("KIS", {"kind": "TEXT", "text": "three"}),
        ("AVS", {"kind": "TEXT", "text": "three"}),
        ("VQA", {"kind": "TEMPORAL", "video_id": "video", "start_ms": 0, "end_ms": 0}),
        ("UNRECOGNIZED", {"kind": "TEMPORAL", "video_id": "video", "start_ms": 0, "end_ms": 0}),
    ],
)
def test_task_kind_mismatch_or_unknown_task_fails_before_dres_submission(
    task_type: str,
    answer: dict[str, Any],
) -> None:
    """Fail closed when live DRES task semantics do not match the answer kind."""

    app, service, state = _harness(task_type=task_type, task_group=task_type)
    try:
        with TestClient(app) as client:
            _connect(client)
            scope = _current_task(client)
            response = client.post("/api/v1/vbs/submit", json={
                "user_id": "member-1",
                "expected_task_scope_key": scope["task_scope_key"],
                "answer": answer,
            })

        assert response.status_code == 422
        assert response.json()["detail"]["code"] in {
            "ANSWER_KIND_MISMATCH",
            "UNSUPPORTED_DRES_TASK_TYPE",
        }
        assert state["submissions"] == []
    finally:
        asyncio.run(service.aclose())


def test_stale_expected_task_scope_returns_structured_409_without_submission() -> None:
    """Reject a frozen popup scope after DRES changes tasks without posting an answer."""

    app, service, state = _harness()
    try:
        with TestClient(app) as client:
            _connect(client)
            old_scope = _current_task(client)
            state["task"] = {
                "name": "VQA task",
                "taskGroup": "VQA",
                "taskType": "VQA",
                "duration": 300,
            }

            response = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(old_scope),
            )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "TASK_SCOPE_MISMATCH"
        assert state["submissions"] == []
    finally:
        asyncio.run(service.aclose())


@pytest.mark.parametrize(
    "status_code,expected_kind,expected_recorded,expected_reason",
    [
        (400, "NOT_RECORDED", False, "DRES_REJECTED"),
        (401, "NOT_RECORDED", False, "DRES_AUTH_REJECTED"),
        (503, "UNKNOWN", None, None),
    ],
)
def test_submission_outcome_is_structured_and_never_replayed(
    status_code: int,
    expected_kind: str,
    expected_recorded: bool | None,
    expected_reason: str | None,
) -> None:
    """Return definitive/ambiguous outcomes over 2xx and make only one DRES POST."""

    app, service, state = _harness(submit_statuses=[status_code, 202])
    try:
        with TestClient(app) as client:
            _connect(client)
            scope = _current_task(client)
            response = client.post("/api/v1/vbs/submit", json=_temporal_request(scope))
            session_status = client.get("/api/v1/vbs/session/member-1").json()

        assert response.status_code == 200, response.text
        result = response.json()
        assert result["state"] == expected_kind
        assert result["recorded"] is expected_recorded
        assert result["verdict"] is None
        if expected_reason is not None:
            assert result["reason"] == expected_reason
        assert len(state["submissions"]) == 1
        assert state["logins"]["dres-member-1"] == 1
        if status_code == 401:
            assert session_status["connected"] is False
        else:
            assert session_status["connected"] is True
    finally:
        asyncio.run(service.aclose())


def test_success_records_each_official_verdict_without_treating_wrong_as_rejection() -> None:
    """DRES delivery status controls recording; the judge verdict is independent."""

    for verdict in ("CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"):
        app, service, state = _harness(submit_verdict=verdict)
        try:
            with TestClient(app) as client:
                _connect(client)
                scope = _current_task(client)
                response = client.post(
                    "/api/v1/vbs/submit",
                    json=_temporal_request(scope),
                )

            assert response.status_code == 200, response.text
            assert response.json()["state"] == "RECORDED"
            assert response.json()["recorded"] is True
            assert response.json()["verdict"] == verdict
            assert len(state["submissions"]) == 1
        finally:
            asyncio.run(service.aclose())


def test_malformed_dres_success_body_returns_unknown_without_replay() -> None:
    """Do not claim delivery when DRES accepts HTTP but omits its verdict."""

    app, service, state = _harness(malformed_success=True)
    try:
        with TestClient(app) as client:
            _connect(client)
            scope = _current_task(client)
            response = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(scope),
            )

        assert response.status_code == 200
        assert response.json()["state"] == "UNKNOWN"
        assert response.json()["recorded"] is None
        assert response.json()["verdict"] is None
        assert len(state["submissions"]) == 1
    finally:
        asyncio.run(service.aclose())


def test_two_participants_submit_through_distinct_private_sessions() -> None:
    """Keep the user-ID to private DRES session mapping isolated per participant."""

    app, service, state = _harness(submit_statuses=[202, 202])
    try:
        with TestClient(app) as client:
            for user_id in ("member-1", "member-2"):
                _connect(client, user_id)
                scope = _current_task(client, user_id)
                response = client.post(
                    "/api/v1/vbs/submit",
                    json=_temporal_request(scope, user_id=user_id, video_id=user_id),
                )
                assert response.status_code == 200, response.text

        assert [session for session, _body in state["submissions"]] == [
            "private-session-member-1",
            "private-session-member-2",
        ]
        assert state["logins"] == {
            "dres-member-1": 1,
            "dres-member-2": 1,
        }
        assert all(
            b"private-session" not in body
            and b"backend-password" not in body
            for _session, body in state["submissions"]
        )
    finally:
        asyncio.run(service.aclose())


def test_submission_401_evicts_only_that_user_and_requires_explicit_reconnect() -> None:
    """Expire one cached identity without refreshing or disrupting another user."""

    app, service, state = _harness(submit_statuses=[401, 202, 202])
    try:
        with TestClient(app) as client:
            _connect(client, "member-1")
            _connect(client, "member-2")
            scope_1 = _current_task(client, "member-1")
            first = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(scope_1, user_id="member-1"),
            )
            second_user_status = client.get("/api/v1/vbs/session/member-2").json()
            no_reconnect = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(scope_1, user_id="member-1"),
            )
            scope_2 = _current_task(client, "member-2")
            second_user_submit = client.post(
                "/api/v1/vbs/submit",
                json=_temporal_request(scope_2, user_id="member-2"),
            )

        assert first.status_code == 200
        assert first.json()["state"] == "NOT_RECORDED"
        assert first.json()["reason"] == "DRES_AUTH_REJECTED"
        assert second_user_status["connected"] is True
        assert no_reconnect.status_code == 401
        assert no_reconnect.json()["detail"]["code"] == "PARTICIPANT_NOT_CONNECTED"
        assert second_user_submit.status_code == 200
        assert [session for session, _body in state["submissions"]] == [
            "private-session-member-1",
            "private-session-member-2",
        ]
        assert state["logins"] == {
            "dres-member-1": 1,
            "dres-member-2": 1,
        }
    finally:
        asyncio.run(service.aclose())


def test_missing_dres_configuration_has_stable_structured_error() -> None:
    """Keep the direct route's configuration failure browser-readable."""

    app = FastAPI()
    app.include_router(create_vbs_router({"vbs_service": None}))
    with TestClient(app) as client:
        response = client.post("/api/v1/vbs/submit", json={
            "user_id": "member-1",
            "expected_task_scope_key": "scope",
            "answer": {"kind": "TEXT", "text": "answer"},
        })

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DRES_NOT_CONFIGURED"
