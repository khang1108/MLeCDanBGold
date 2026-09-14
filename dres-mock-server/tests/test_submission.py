"""Contract tests for DRES login, discovery, and submission models."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from dres_mock_server.models import (
    ApiClientAnswer,
    ApiClientAnswerSet,
    ApiClientEvaluationInfo,
    ApiClientSubmission,
    ApiClientTaskTemplateInfo,
    ApiUser,
    DresVerdict,
    LoginRequest,
    SuccessfulSubmissionsStatus,
)


def test_login_and_discovery_models_use_official_aliases() -> None:
    """Required aliases serialize with DRES casing and current task has no ID."""

    login = LoginRequest(username="member-1", password="password-1")
    task = ApiClientTaskTemplateInfo(
        name="KIS task",
        taskGroup="KIS",
        taskType="KIS",
        duration=300,
    )
    evaluation = ApiClientEvaluationInfo(
        id="eval-local",
        name="Local evaluation",
        type="SYNCHRONOUS",
        status="ACTIVE",
        templateId="template-1",
        teams=["team-1"],
        taskTemplates=[task],
    )
    user = ApiUser(username="member-1", sessionId="opaque-session")

    assert login.model_dump() == {"username": "member-1", "password": "password-1"}
    assert task.model_dump(by_alias=True) == {
        "name": "KIS task",
        "taskGroup": "KIS",
        "taskType": "KIS",
        "duration": 300,
    }
    assert "taskId" not in task.model_dump(by_alias=True)
    assert evaluation.model_dump(by_alias=True)["taskTemplates"][0]["taskGroup"] == "KIS"
    assert user.model_dump(by_alias=True, exclude_none=True) == {
        "username": "member-1",
        "sessionId": "opaque-session",
    }
    with pytest.raises(ValidationError):
        ApiClientTaskTemplateInfo(
            name="KIS task",
            taskGroup="KIS",
            taskType="KIS",
            taskId="not-part-of-current-task-response",
        )


def test_login_discovery_and_submission_forbid_unknown_or_mis_cased_fields() -> None:
    """Closed schemas reject misspelled aliases and undeclared properties."""

    with pytest.raises(ValidationError):
        LoginRequest(username="member-1", password="secret", extra="x")
    with pytest.raises(ValidationError):
        ApiClientTaskTemplateInfo(
            name="KIS task",
            taskgroup="KIS",
            taskType="KIS",
        )
    with pytest.raises(ValidationError):
        ApiClientSubmission.model_validate(
            {"answerSets": [{"answers": [{"mediaitemName": "00001"}]}]}
        )


@pytest.mark.parametrize(
    "answer",
    [
        {"mediaItemName": "00001", "start": 1234, "end": 1234},
        {"mediaItemName": "00001", "start": 1234, "end": 5678},
        {"text": "three"},
        {"mediaItemName": "00001"},
    ],
)
def test_submission_accepts_supported_answer_forms(answer: dict[str, object]) -> None:
    """Text, item, temporal point, and temporal range forms are accepted."""

    submission = ApiClientSubmission(answerSets=[{"answers": [answer]}])

    assert len(submission.answer_sets) == 1
    assert len(submission.answer_sets[0].answers) == 1


def test_submission_preserves_aliases_order_and_optional_task_metadata() -> None:
    """Answer ordering and original media identifiers survive round trips."""

    submission = ApiClientSubmission.model_validate(
        {
            "answerSets": [
                {
                    "taskId": "optional-id",
                    "taskName": "AVS task",
                    "answers": [
                        {"mediaItemName": "00002", "start": 2000, "end": 2500},
                        {"mediaItemName": "00001", "start": 1000, "end": 1500},
                    ],
                }
            ]
        }
    )

    assert submission.model_dump(by_alias=True, exclude_none=True) == {
        "answerSets": [
            {
                "taskId": "optional-id",
                "taskName": "AVS task",
                "answers": [
                    {"mediaItemName": "00002", "start": 2000, "end": 2500},
                    {"mediaItemName": "00001", "start": 1000, "end": 1500},
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"text": "   "},
        {"mediaItemName": "  "},
        {"text": "yes", "mediaItemName": "00001"},
        {"text": "yes", "start": 1, "end": 2},
        {"mediaItemName": "00001", "start": 1},
        {"mediaItemName": "00001", "end": 2},
        {"mediaItemName": "00001", "start": -1, "end": 0},
        {"mediaItemName": "00001", "start": 0, "end": -1},
        {"mediaItemName": "00001", "start": 4, "end": 3},
        {"mediaItemName": "00001", "start": "1", "end": 2},
        {"text": "yes", "extra": True},
    ],
)
def test_submission_rejects_invalid_answers(answer: dict[str, object]) -> None:
    """Answer validators enforce one usable form and complete integer ranges."""

    with pytest.raises(ValidationError):
        ApiClientAnswer.model_validate(answer)


def test_submission_requires_nonempty_answer_sets_and_answers() -> None:
    """DRES requires at least one set and at least one answer in each set."""

    with pytest.raises(ValidationError):
        ApiClientSubmission(answerSets=[])
    with pytest.raises(ValidationError):
        ApiClientSubmission(answerSets=[{"answers": []}])


def test_submission_verdict_uses_the_official_enum_values() -> None:
    """The verdict response is constrained to DRES's four declared values."""

    response = SuccessfulSubmissionsStatus(
        status=True,
        submission=DresVerdict.INDETERMINATE,
        description="mock accepted",
    )

    assert response.model_dump() == {
        "status": True,
        "submission": DresVerdict.INDETERMINATE,
        "description": "mock accepted",
    }
    with pytest.raises(ValidationError):
        SuccessfulSubmissionsStatus(
            status=True,
            submission="MAYBE",
            description="mock accepted",
        )


def test_kis_submission_returns_the_default_indeterminate_verdict(
    client: TestClient,
    session_id: str,
) -> None:
    """One KIS temporal point is accepted without semantic scoring."""

    response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={
            "answerSets": [
                {
                    "taskName": "KIS task",
                    "answers": [
                        {"mediaItemName": "video-00001", "start": 1234, "end": 1234}
                    ],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": True,
        "submission": "INDETERMINATE",
        "description": "mock accepted",
    }
    record = client.app.state.mock_state.captured_requests("SUBMISSION")[0]
    assert record.payload["answerSets"][0]["answers"][0]["mediaItemName"] == "video-00001"
    assert record.payload["answerSets"][0]["answers"][0]["start"] == 1234
    assert record.payload["answerSets"][0]["answers"][0]["end"] == 1234


def test_vqa_text_answer_and_media_range_are_accepted(
    client: TestClient,
    session_id: str,
) -> None:
    """The mock accepts both VQA text and DRES media ranges."""

    text_response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={"answerSets": [{"taskName": "KIS task", "answers": [{"text": "three"}]}]},
    )
    range_response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={
            "answerSets": [
                {
                    "taskName": "KIS task",
                    "answers": [{"mediaItemName": "clip-2", "start": 100, "end": 900}],
                }
            ]
        },
    )

    assert text_response.status_code == range_response.status_code == 200
    assert len(client.app.state.mock_state.captured_requests("SUBMISSION")) == 2


def test_avs_ordered_answers_are_kept_in_one_submission(
    client: TestClient,
    session_id: str,
) -> None:
    """A multi-answer AVS set stays in one request and preserves answer order."""

    task = ApiClientTaskTemplateInfo(
        name="AVS task",
        taskGroup="AVS",
        taskType="AVS",
        duration=300,
    )
    client.app.state.mock_state.set_task(active=True, task=task)
    answers = [
        {"mediaItemName": "00002", "start": 2000, "end": 2000},
        {"mediaItemName": "00001", "start": 1000, "end": 1000},
    ]

    response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={"answerSets": [{"taskName": "AVS task", "answers": answers}]},
    )

    assert response.status_code == 200
    records = client.app.state.mock_state.captured_requests("SUBMISSION")
    assert len(records) == 1
    assert records[0].payload["answerSets"][0]["answers"] == answers


def test_submission_can_omit_task_name_or_match_current_task(
    client: TestClient,
    session_id: str,
) -> None:
    """DRES may infer an omitted task name and accepts an explicit match."""

    omitted = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={"answerSets": [{"answers": [{"text": "three"}]}]},
    )
    matching = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={
            "answerSets": [
                {"taskName": "KIS task", "answers": [{"text": "four"}]}
            ]
        },
    )

    assert omitted.status_code == matching.status_code == 200


def test_stale_task_name_is_captured_as_a_precondition_rejection(
    client: TestClient,
    session_id: str,
) -> None:
    """Structurally valid stale task submissions are captured with HTTP 412."""

    response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={
            "answerSets": [
                {"taskName": "stale task", "answers": [{"text": "three"}]}
            ]
        },
    )

    assert response.status_code == 412
    assert response.json() == {
        "status": False,
        "description": "Submission task name does not match the active task.",
    }
    record = client.app.state.mock_state.captured_requests("SUBMISSION")[0]
    assert record.outcome.status_code == 412
    assert record.outcome.verdict is None


def test_explicit_task_id_is_rejected_because_mock_has_no_instance_id(
    client: TestClient,
    session_id: str,
) -> None:
    """The local mock never invents or matches a DRES task instance ID."""

    response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json={
            "answerSets": [
                {
                    "taskId": "invented-id",
                    "taskName": "KIS task",
                    "answers": [{"text": "three"}],
                }
            ]
        },
    )

    assert response.status_code == 412
    assert response.json()["status"] is False
    assert client.app.state.mock_state.captured_requests("SUBMISSION")[0].outcome.status_code == 412


@pytest.mark.parametrize(
    "payload",
    [
        {"answerSets": []},
        {"answerSets": [{"answers": []}]},
        {"answerSets": [{"answers": [{"text": "  "}]}]},
        {"answerSets": [{"answers": [{"mediaItemName": "x", "start": -1, "end": 0}]}]},
        {"answerSets": [{"answers": [{"mediaItemName": "x", "start": 3, "end": 2}]}]},
        {"answerSets": [{"answers": [{"text": "ok", "unexpected": True}]}]},
    ],
)
def test_invalid_submission_bodies_return_dres_400_without_capture(
    client: TestClient,
    session_id: str,
    payload: dict[str, object],
) -> None:
    """Only structurally and semantically valid submissions enter the journal."""

    response = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": session_id},
        json=payload,
    )

    assert response.status_code == 400
    assert response.json()["status"] is False
    assert client.app.state.mock_state.captured_requests("SUBMISSION") == []


def test_invalid_session_and_wrong_evaluation_are_not_captured(
    client: TestClient,
) -> None:
    """Authentication and evaluation failures leave the request journal alone."""

    payload = {"answerSets": [{"answers": [{"text": "three"}]}]}
    invalid_session = client.post(
        "/api/v2/submit/eval-vbs-local",
        params={"session": "expired-token"},
        json=payload,
    )
    valid = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": "password-1"},
    ).json()["sessionId"]
    wrong_evaluation = client.post(
        "/api/v2/submit/other-evaluation",
        params={"session": valid},
        json=payload,
    )

    assert invalid_session.status_code == 401
    assert wrong_evaluation.status_code == 404
    assert client.app.state.mock_state.captured_requests("SUBMISSION") == []
