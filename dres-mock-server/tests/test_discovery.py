"""Black-box tests for DRES evaluation and active-task discovery."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_evaluation_list_contains_one_active_synchronous_run(
    client: TestClient,
    session_id: str,
) -> None:
    """The run summary contains the required client metadata and one task."""

    response = client.get(
        "/api/v2/client/evaluation/list",
        params={"session": session_id},
    )

    assert response.status_code == 200
    evaluations = response.json()
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    assert evaluation["id"] == "eval-vbs-local"
    assert evaluation["type"] == "SYNCHRONOUS"
    assert evaluation["status"] == "ACTIVE"
    assert evaluation["taskTemplates"] == [
        {
            "name": "KIS task",
            "taskGroup": "KIS",
            "taskType": "KIS",
            "duration": 300,
        }
    ]


def test_current_task_contains_template_metadata_but_no_task_id(
    client: TestClient,
    session_id: str,
) -> None:
    """Current-task discovery returns only official task template fields."""

    response = client.get(
        "/api/v2/client/evaluation/currentTask/eval-vbs-local",
        params={"session": session_id},
    )

    assert response.status_code == 200
    assert response.json() == {
        "name": "KIS task",
        "taskGroup": "KIS",
        "taskType": "KIS",
        "duration": 300,
    }
    assert "taskId" not in response.json()


def test_current_task_for_wrong_evaluation_is_not_found(
    client: TestClient,
    session_id: str,
) -> None:
    """The mock serves metadata only for its configured evaluation."""

    response = client.get(
        "/api/v2/client/evaluation/currentTask/other-evaluation",
        params={"session": session_id},
    )

    assert response.status_code == 404
    assert response.json()["status"] is False


def test_missing_active_task_returns_not_found(
    client: TestClient,
    session_id: str,
) -> None:
    """The local test control can disable the active-task response."""

    state = client.app.state.mock_state
    state.set_active_task(None)

    response = client.get(
        "/api/v2/client/evaluation/currentTask/eval-vbs-local",
        params={"session": session_id},
    )

    assert response.status_code == 404
    assert response.json()["status"] is False
