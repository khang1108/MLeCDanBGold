"""Black-box tests for local-only task, scenario, reset, and state controls."""

from __future__ import annotations

import logging
import importlib

from fastapi.testclient import TestClient

from dres_mock_server.settings import MockSettings


def _submission(task_name: str | None = "KIS task") -> dict[str, object]:
    """Return one valid temporal answer set for exercising the HTTP route."""

    answer_set: dict[str, object] = {
        "answers": [{"mediaItemName": "00001", "start": 1234, "end": 1234}]
    }
    if task_name is not None:
        answer_set["taskName"] = task_name
    return {"answerSets": [answer_set]}


def _result_log() -> dict[str, object]:
    """Return one complete but hand-checkable official result log."""

    return {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 1, "answer": {"mediaItemName": "00001", "start": 1234, "end": 1234}}
        ],
        "events": [
            {
                "timestamp": 1_800_000_000_000,
                "category": "TEXT",
                "type": "SEARCH",
                "value": "person running",
            }
        ],
    }


def test_state_control_contains_counts_and_never_exposes_secrets(
    client: TestClient,
    session_id: str,
) -> None:
    """The operator view contains captured metadata but no password or token."""

    client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )
    response = client.get("/__test/state")
    state = response.json()

    assert response.status_code == 200
    assert state["evaluation"]["id"] == "eval-vbs-local"
    assert state["task"]["active"] is True
    assert state["session_count"] == 1
    assert state["submission_count"] == 1
    assert state["result_log_count"] == 0
    assert state["submissions"][0]["username"] == "member-1"
    assert state["submissions"][0]["session_fingerprint"] != session_id
    assert len(state["submissions"][0]["session_fingerprint"]) == 12
    assert session_id not in response.text
    assert "password-1" not in response.text
    assert "sessionId" not in response.text


def test_state_control_and_captured_repr_do_not_leak_password_or_token(
    client: TestClient,
    session_id: str,
    caplog: logging.LogCaptureFixture,
) -> None:
    """Application logs and journal records avoid login secrets and session IDs."""

    caplog.set_level(logging.INFO)
    client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )
    record = client.app.state.mock_state.captured_requests("SUBMISSION")[0]

    assert session_id not in repr(record)
    assert "password-1" not in repr(record)
    app_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name.startswith(("dres_mock_server", "uvicorn.access"))
    )
    assert session_id not in app_logs
    assert "password-1" not in app_logs


def test_task_control_switches_and_disables_current_task(
    client: TestClient,
    session_id: str,
) -> None:
    """Operator task changes persist and the current-task route follows them."""

    changed = client.put(
        "/__test/task",
        json={
            "active": True,
            "name": "AVS task",
            "taskGroup": "AVS",
            "taskType": "AVS",
            "duration": 240,
        },
    )
    active = client.get(
        "/api/v2/client/evaluation/currentTask/eval-vbs-local",
        params={"session": session_id},
    )
    disabled = client.put(
        "/__test/task",
        json={
            "active": False,
            "name": "AVS task",
            "taskGroup": "AVS",
            "taskType": "AVS",
            "duration": 240,
        },
    )
    unavailable = client.get(
        "/api/v2/client/evaluation/currentTask/eval-vbs-local",
        params={"session": session_id},
    )

    assert changed.status_code == 200
    assert active.json() == {
        "name": "AVS task",
        "taskGroup": "AVS",
        "taskType": "AVS",
        "duration": 240,
    }
    assert disabled.status_code == 200
    assert unavailable.status_code == 404
    assert client.get("/__test/state").json()["task"]["active"] is False


def test_task_control_validation_returns_dres_400_without_reflecting_values(
    client: TestClient,
) -> None:
    """Malformed control data is rejected without exposing arbitrary input."""

    secret = "control-value-that-must-not-echo"
    response = client.put(
        "/__test/task",
        json={
            "active": True,
            "name": " ",
            "taskGroup": "KIS",
            "taskType": "KIS",
            "duration": 300,
            "unexpected": secret,
        },
    )

    assert response.status_code == 400
    assert response.json()["status"] is False
    assert secret not in response.text


def test_submission_scenario_is_consumed_once_by_a_valid_request(
    client: TestClient,
    session_id: str,
) -> None:
    """One accepted request uses the configured response; the next uses defaults."""

    configured = client.put(
        "/__test/scenario/submission",
        json={
            "statusCode": 202,
            "verdict": "WRONG",
            "delayMs": 0,
            "malformedBody": False,
            "description": "queued for judging",
        },
    )
    first = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )
    second = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )

    assert configured.status_code == 200
    assert first.status_code == 202
    assert first.json() == {
        "status": True,
        "submission": "WRONG",
        "description": "queued for judging",
    }
    assert second.status_code == 200
    assert second.json()["submission"] == "INDETERMINATE"
    state = client.get("/__test/state").json()
    assert [record["outcome"]["status_code"] for record in state["submissions"]] == [200, 202]
    assert state["scenarios"]["submission"]["statusCode"] == 200


def test_invalid_and_task_rejected_submissions_do_not_consume_scenario(
    client: TestClient,
    session_id: str,
) -> None:
    """Only a valid request for the active task consumes the one-shot response."""

    client.put(
        "/__test/scenario/submission",
        json={"statusCode": 202, "verdict": "WRONG", "description": "one shot"},
    )
    invalid_body = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={"answerSets": [{"answers": []}]},
    )
    stale_task = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission("old task"),
    )
    pending = client.get("/__test/state").json()["scenarios"]["submission"]
    valid = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )

    assert invalid_body.status_code == 400
    assert stale_task.status_code == 412
    assert pending["statusCode"] == 202
    assert valid.status_code == 202
    records = client.get("/__test/state").json()["submissions"]
    assert records[1]["outcome"]["status_code"] == 412
    assert records[0]["outcome"]["status_code"] == 202


def test_result_log_scenario_is_separate_and_returns_only_success_status(
    client: TestClient,
    session_id: str,
) -> None:
    """Result-log failures do not consume the submission scenario and vice versa."""

    client.put(
        "/__test/scenario/submission",
        json={"statusCode": 202, "verdict": "INDETERMINATE", "description": "later"},
    )
    client.put(
        "/__test/scenario/result-log",
        json={"statusCode": 500, "description": "log storage unavailable"},
    )
    failed_log = client.post(
        "/api/v2/log/result/eval-vbs-local",
        params={"session": session_id},
        json=_result_log(),
    )
    accepted_submission = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission(),
    )

    assert failed_log.status_code == 500
    assert failed_log.json() == {
        "status": False,
        "description": "log storage unavailable",
    }
    assert accepted_submission.status_code == 202
    state = client.get("/__test/state").json()
    assert state["result_logs"][0]["outcome"]["status_code"] == 500
    assert state["submissions"][0]["outcome"]["status_code"] == 202


def test_reset_restores_task_scenarios_sessions_and_empty_journals(
    client: TestClient,
    session_id: str,
) -> None:
    """Reset starts a clean local run with the documented defaults."""

    client.put(
        "/__test/task",
        json={"active": True, "name": "VQA task", "taskGroup": "VQA", "taskType": "VQA", "duration": 60},
    )
    client.put(
        "/__test/scenario/submission",
        json={"statusCode": 500, "verdict": "WRONG", "description": "reset this"},
    )
    client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=_submission("VQA task"),
    )

    response = client.post("/__test/reset")
    state = client.get("/__test/state").json()

    assert response.status_code == 200
    assert state["session_count"] == 0
    assert state["submission_count"] == state["result_log_count"] == 0
    assert state["submissions"] == state["result_logs"] == []
    assert state["task"] == {
        "name": "KIS task",
        "taskGroup": "KIS",
        "taskType": "KIS",
        "duration": 300,
        "active": True,
    }
    assert state["scenarios"]["submission"]["statusCode"] == 200


def test_test_controls_are_tagged_and_all_http_routes_have_mock_header(
    client: TestClient,
) -> None:
    """OpenAPI warns operators and all delivered responses carry the mock marker."""

    openapi = client.get("/openapi.json").json()
    control_tag = next(tag for tag in openapi["tags"] if tag["name"] == "Test Control")
    assert "TEST CONTROL — LOCAL ONLY" in control_tag["description"]

    for method, path in [("get", "/__test/state"), ("post", "/__test/reset"), ("put", "/__test/task")]:
        operation = openapi["paths"][path][method]
        assert "Test Control" in operation["tags"]

    for response in [client.get("/__test/state"), client.get("/docs"), client.get("/openapi.json")]:
        assert response.headers["X-DRES-Mock"] == "true"


def test_console_server_disables_access_logs_for_session_query_strings(
    monkeypatch,
) -> None:
    """The real Uvicorn entry point cannot log DRES session query parameters."""

    module = importlib.import_module("dres_mock_server.app")
    captured: dict[str, object] = {}
    monkeypatch.setattr(module, "MockSettings", lambda: MockSettings(_env_file=None))
    monkeypatch.setattr(module.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))

    module.run()

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8090
    assert captured["access_log"] is False
