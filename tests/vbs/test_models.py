"""Strict wire-contract tests for the subset of DRES v2 HCMAI consumes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hcmai.vbs.models import (
    ApiClientAnswer,
    ApiClientAnswerSet,
    ApiClientSubmission,
    DresEvaluation,
    DresSubmissionStatus,
    DresStatus,
    DresTaskTemplateInfo,
    DresUser,
    QueryEvent,
    QueryResultLog,
)


def test_temporal_and_text_submission_payloads_use_dres_casing() -> None:
    """Serialize only the official DRES client fields and aliases."""

    payload = ApiClientSubmission(answer_sets=[
        ApiClientAnswerSet(
            task_name="KIS task",
            answers=[
                ApiClientAnswer(media_item_name="00001", start=12345, end=12345),
                ApiClientAnswer(text="a red car"),
            ],
        ),
    ])

    assert payload.model_dump(by_alias=True, exclude_none=True) == {
        "answerSets": [{
            "taskName": "KIS task",
            "answers": [
                {"mediaItemName": "00001", "start": 12345, "end": 12345},
                {"text": "a red car"},
            ],
        }],
    }


def test_query_result_log_contains_events_and_complete_ranked_answer_shape() -> None:
    """Keep event time separate from media timeline coordinates."""

    payload = QueryResultLog(
        timestamp=1_800_000_000_000,
        sort_type="list",
        result_set_availability="",
        results=[
            {"rank": 1, "answer": {"mediaItemName": "video-a", "start": 12000, "end": 12000}},
        ],
        events=[QueryEvent(
            timestamp=1_800_000_000_000,
            category="TEXT",
            type="SEARCH",
            value="  a person running  ",
        )],
    )

    assert payload.model_dump(by_alias=True, exclude_none=True) == {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [{
            "rank": 1,
            "answer": {"mediaItemName": "video-a", "start": 12000, "end": 12000},
        }],
        "events": [{
            "timestamp": 1_800_000_000_000,
            "category": "TEXT",
            "type": "SEARCH",
            "value": "  a person running  ",
        }],
    }


def test_minimal_dres_responses_validate_exact_fields() -> None:
    """Parse the minimal login, evaluation, current-task, and status contracts."""

    user = DresUser.model_validate({
        "id": "participant-1",
        "username": "team-a",
        "role": "PARTICIPANT",
        "sessionId": "session-secret",
    })
    evaluation = DresEvaluation.model_validate({
        "id": "eval-4",
        "name": "VBS test",
        "type": "SYNCHRONOUS",
        "status": "ACTIVE",
        "templateId": "template-1",
        "teams": ["team-a"],
        "taskTemplates": [{"name": "KIS", "taskGroup": "KIS", "taskType": "KIS"}],
    })
    task = DresTaskTemplateInfo.model_validate({
        "name": "KIS task",
        "taskGroup": "KIS",
        "taskType": "KIS",
        "duration": 300,
    })
    status = DresStatus.model_validate({"status": True, "description": "accepted"})

    assert user.session_id == "session-secret"
    assert evaluation.status == "ACTIVE"
    assert task.name == "KIS task"
    assert task.task_group == "KIS"
    assert task.task_type == "KIS"
    assert task.duration == 300
    assert status.status is True


def test_current_task_rejects_nonexistent_instantiated_task_id() -> None:
    """Reject task IDs because the DRES current-task schema has no such field."""

    with pytest.raises(ValidationError):
        DresTaskTemplateInfo.model_validate({
            "taskId": "invented-id",
            "name": "KIS",
            "taskGroup": "KIS",
            "taskType": "KIS",
        })


def test_current_task_validates_nonblank_fields_without_rewriting_them() -> None:
    """Preserve task text for stable scope hashing and exact taskName submits."""

    task = DresTaskTemplateInfo.model_validate({
        "name": " KIS task ",
        "taskGroup": " KIS ",
        "taskType": " KIS ",
        "duration": 300,
    })

    assert task.name == " KIS task "
    assert task.task_group == " KIS "
    assert task.task_type == " KIS "

    with pytest.raises(ValidationError):
        DresTaskTemplateInfo.model_validate({
            "name": "   ",
            "taskGroup": "KIS",
            "taskType": "KIS",
        })


def test_successful_submission_response_requires_and_preserves_verdict() -> None:
    """Keep delivery acceptance separate from DRES correctness verdicts."""

    result = DresSubmissionStatus.model_validate({
        "status": True,
        "submission": "CORRECT",
        "description": "accepted",
    })

    assert result.status is True
    assert result.submission == "CORRECT"
    assert result.description == "accepted"

    with pytest.raises(ValidationError):
        DresSubmissionStatus.model_validate({
            "status": True,
            "description": "missing verdict",
        })
    with pytest.raises(ValidationError):
        DresSubmissionStatus.model_validate({
            "status": True,
            "submission": "MAYBE",
            "description": "unknown verdict",
        })


@pytest.mark.parametrize(
    "model,payload",
    [
        (ApiClientAnswer, {"frame_idx": 1}),
        (ApiClientAnswerSet, {"answers": [], "taskId": "invented-id"}),
        (ApiClientSubmission, {"answerSets": [], "extra": True}),
        (QueryEvent, {"timestamp": 1, "category": "TEXT", "type": "SEARCH"}),
        (QueryResultLog, {"timestamp": 1, "sortType": "list", "resultSetAvailability": "", "results": [], "events": [], "extra": 1}),
    ],
)
def test_dres_contracts_forbid_unknown_or_missing_fields(model, payload) -> None:
    """Catch accidental internal fields and incomplete DRES JSON at the edge."""

    with pytest.raises(ValidationError):
        model.model_validate(payload)
