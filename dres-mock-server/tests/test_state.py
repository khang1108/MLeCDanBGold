"""Unit tests for synchronized sessions, scenarios, and bounded journals."""

from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest

from dres_mock_server.models import (
    ApiClientSubmission,
    ApiClientTaskTemplateInfo,
    DresVerdict,
    QueryResultLog,
)
from dres_mock_server.settings import MockSettings
from dres_mock_server.state import MockState, ResponseScenario


def _state() -> MockState:
    """Build deterministic local settings without reading ambient environment."""

    settings = MockSettings(
        _env_file=None,
        users={"member-1": "password-1", "member-2": "password-2"},
    )
    return MockState(settings)


def _submission(*, task_name: str | None = None, include_task_name: bool = False) -> ApiClientSubmission:
    """Build one simple media-point submission with optional task metadata."""

    answer_set: dict[str, object] = {
        "answers": [{"mediaItemName": "00001", "start": 1234, "end": 1234}]
    }
    if include_task_name:
        answer_set["taskName"] = task_name
    return ApiClientSubmission.model_validate({"answerSets": [answer_set]})


def _result_log() -> QueryResultLog:
    """Build a valid empty result log for state-only tests."""

    return QueryResultLog(
        timestamp=1_800_000_000_000,
        sortType="list",
        resultSetAvailability="",
        results=[],
        events=[],
    )


def test_login_creates_opaque_sessions_and_rejects_bad_credentials() -> None:
    """Valid credentials get distinct tokens; invalid login changes no state."""

    state = _state()

    assert state.login("member-1", "wrong-password") is None
    assert state.session_count == 0

    token_one = state.login("member-1", "password-1")
    token_two = state.login("member-1", "password-1")

    assert token_one is not None
    assert token_two is not None
    assert token_one != token_two
    assert len(token_one) >= 32
    assert state.session_username(token_one) == "member-1"
    assert state.session_username("unknown-session") is None
    assert state.session_count == 2
    assert token_one not in repr(state)
    assert "password-1" not in repr(state)


def test_logout_invalidates_only_the_selected_session() -> None:
    """Logging out one session leaves other sessions for the same user active."""

    state = _state()
    token_one = state.login("member-1", "password-1")
    token_two = state.login("member-1", "password-1")
    assert token_one is not None and token_two is not None

    assert state.logout(token_one) is True
    assert state.logout(token_one) is False
    assert state.session_username(token_one) is None
    assert state.session_username(token_two) == "member-1"
    assert state.session_count == 1


def test_captures_have_global_ids_safe_fingerprints_and_exact_payload_fields() -> None:
    """Records preserve aliases and explicit nulls without revealing secrets."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None

    omitted = state.capture_submission(token, "eval-vbs-local", _submission())
    explicit_null = state.capture_submission(
        token,
        "eval-vbs-local",
        _submission(task_name=None, include_task_name=True),
    )
    result_log = state.capture_result_log(token, "eval-vbs-local", _result_log())
    assert omitted is not None and explicit_null is not None and result_log is not None

    assert [omitted.record_id, explicit_null.record_id, result_log.record_id] == [1, 2, 3]
    assert omitted.kind == "SUBMISSION"
    assert result_log.kind == "RESULT_LOG"
    assert omitted.username == "member-1"
    assert omitted.session_fingerprint == hashlib.sha256(token.encode()).hexdigest()[:12]
    assert token not in repr(omitted)
    assert "password-1" not in repr(omitted)
    assert "session_id" not in repr(omitted)

    omitted_answer_set = omitted.payload["answerSets"][0]
    explicit_answer_set = explicit_null.payload["answerSets"][0]
    assert "taskName" not in omitted_answer_set
    assert explicit_answer_set["taskName"] is None
    assert "mediaItemName" in omitted_answer_set["answers"][0]
    assert "media_item_name" not in omitted_answer_set["answers"][0]
    assert result_log.payload["resultSetAvailability"] == ""
    assert result_log.payload["results"] == []
    assert result_log.payload["events"] == []


def test_task_resolution_rejections_are_recorded_without_consuming_scenario() -> None:
    """No task, explicit task IDs, and stale names return captured 412 outcomes."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None
    one_shot = ResponseScenario(
        status_code=202,
        verdict=DresVerdict.WRONG,
        description="queued for judging",
    )
    state.set_scenario("SUBMISSION", one_shot)

    stale = state.capture_submission(
        token,
        "eval-vbs-local",
        _submission(task_name="old task", include_task_name=True),
    )
    with_task_id = ApiClientSubmission.model_validate(
        {
            "answerSets": [
                {
                    "taskId": None,
                    "answers": [{"text": "three"}],
                }
            ]
        }
    )
    explicit_id = state.capture_submission(token, "eval-vbs-local", with_task_id)

    assert stale is not None and explicit_id is not None
    assert stale.outcome.status_code == 412
    assert explicit_id.outcome.status_code == 412
    assert state.snapshot()["scenarios"]["submission"] == {
        "statusCode": 202,
        "verdict": "WRONG",
        "delayMs": 0,
        "malformedBody": False,
        "description": "queued for judging",
    }

    matching = state.capture_submission(
        token,
        "eval-vbs-local",
        _submission(task_name="KIS task", include_task_name=True),
    )
    inferred = state.capture_submission(token, "eval-vbs-local", _submission())
    assert matching is not None and inferred is not None
    assert matching.outcome.status_code == 202
    assert matching.outcome.verdict is DresVerdict.WRONG
    assert inferred.outcome.status_code == 200
    assert [record.record_id for record in state.captured_requests()] == [1, 2, 3, 4]


def test_inactive_task_preserves_metadata_and_rejections_do_not_consume() -> None:
    """Disabling a task keeps its settings visible until reset or replacement."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None
    avs_task = ApiClientTaskTemplateInfo(
        name="AVS task",
        taskGroup="AVS",
        taskType="AVS",
        duration=120,
    )
    state.set_task(active=False, task=avs_task)
    state.set_scenario(
        "SUBMISSION",
        ResponseScenario(status_code=500, description="one-shot failure"),
    )

    record = state.capture_submission(token, "eval-vbs-local", _submission())
    snapshot = state.snapshot()

    assert record is not None and record.outcome.status_code == 412
    assert state.active_task() is None
    assert snapshot["task"] == {
        "name": "AVS task",
        "taskGroup": "AVS",
        "taskType": "AVS",
        "duration": 120,
        "active": False,
    }
    assert snapshot["scenarios"]["submission"]["statusCode"] == 500


def test_result_and_submission_scenarios_are_separate_one_shots() -> None:
    """Each request kind consumes only its own scenario then restores defaults."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None
    state.set_scenario(
        "SUBMISSION",
        ResponseScenario(status_code=202, description="submission queued"),
    )
    state.set_scenario(
        "RESULT_LOG",
        ResponseScenario(status_code=500, description="log failed"),
    )

    submit_one = state.capture_submission(token, "eval-vbs-local", _submission())
    log_one = state.capture_result_log(token, "eval-vbs-local", _result_log())
    submit_two = state.capture_submission(token, "eval-vbs-local", _submission())
    log_two = state.capture_result_log(token, "eval-vbs-local", _result_log())

    assert submit_one is not None and submit_two is not None
    assert log_one is not None and log_two is not None
    assert submit_one.outcome.status_code == 202
    assert submit_one.outcome.description == "submission queued"
    assert log_one.outcome.status_code == 500
    assert log_one.outcome.verdict is None
    assert submit_two.outcome.status_code == 200
    assert log_two.outcome.status_code == 200
    assert [record.record_id for record in state.captured_requests()] == [1, 2, 3, 4]


def test_authenticated_capture_is_required_before_recording_or_consuming() -> None:
    """Invalid sessions neither create records nor consume configured responses."""

    state = _state()
    state.set_scenario(
        "RESULT_LOG",
        ResponseScenario(status_code=500, description="pending"),
    )

    assert state.capture_submission("bad-session", "eval-vbs-local", _submission()) is None
    assert state.capture_result_log("bad-session", "eval-vbs-local", _result_log()) is None
    assert state.captured_requests() == []
    assert state.snapshot()["scenarios"]["result_log"]["statusCode"] == 500


def test_reset_restores_defaults_clears_state_and_restarts_record_ids() -> None:
    """Reset clears private sessions and both journals and restores defaults."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None
    state.set_active_task(
        ApiClientTaskTemplateInfo(
            name="VQA task",
            taskGroup="VQA",
            taskType="VQA",
            duration=90,
        )
    )
    state.set_scenario(
        "RESULT_LOG",
        ResponseScenario(status_code=404, description="missing result target"),
    )
    state.capture_submission(token, "eval-vbs-local", _submission())
    state.capture_result_log(token, "eval-vbs-local", _result_log())

    state.reset()
    snapshot = state.snapshot()

    assert state.session_count == 0
    assert state.session_username(token) is None
    assert state.active_task() is not None
    assert state.active_task().name == "KIS task"
    assert snapshot["task"]["taskGroup"] == "KIS"
    assert snapshot["task"]["active"] is True
    assert snapshot["submissions"] == []
    assert snapshot["result_logs"] == []
    assert snapshot["submission_count"] == 0
    assert snapshot["result_log_count"] == 0
    assert snapshot["scenarios"]["result_log"]["statusCode"] == 200

    new_token = state.login("member-1", "password-1")
    assert new_token is not None
    first_record = state.capture_submission(new_token, "eval-vbs-local", _submission())
    assert first_record is not None and first_record.record_id == 1


def test_journals_are_capped_per_kind_and_snapshot_is_newest_first() -> None:
    """Each 1,000-entry journal trims independently but shares one ID sequence."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None

    for _ in range(1_001):
        record = state.capture_submission(token, "eval-vbs-local", _submission())
        assert record is not None
    result_log = state.capture_result_log(token, "eval-vbs-local", _result_log())
    assert result_log is not None

    snapshot = state.snapshot()
    assert len(snapshot["submissions"]) == 1_000
    assert len(snapshot["result_logs"]) == 1
    assert snapshot["submissions"][0]["record_id"] == 1_001
    assert snapshot["submissions"][-1]["record_id"] == 2
    assert snapshot["result_logs"][0]["record_id"] == 1_002
    assert [record.record_id for record in state.captured_requests("SUBMISSION")][:2] == [
        2,
        3,
    ]


def test_nested_payloads_returned_to_callers_are_detached() -> None:
    """Mutating a returned record cannot alter the private journal payload."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None
    record = state.capture_submission(token, "eval-vbs-local", _submission())
    assert record is not None

    record.payload["answerSets"][0]["answers"][0]["mediaItemName"] = "changed"
    stored = state.snapshot()["submissions"][0]

    assert stored["payload"]["answerSets"][0]["answers"][0]["mediaItemName"] == "00001"


def test_concurrent_requests_receive_unique_monotonic_record_ids() -> None:
    """The re-entrant lock keeps authentication, capture, and IDs atomic."""

    state = _state()
    token = state.login("member-1", "password-1")
    assert token is not None

    def capture_one(_: int) -> int:
        record = state.capture_submission(token, "eval-vbs-local", _submission())
        assert record is not None
        return record.record_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        ids = list(executor.map(capture_one, range(64)))

    assert sorted(ids) == list(range(1, 65))
    assert [record.record_id for record in state.captured_requests()] == list(range(1, 65))


def test_response_scenario_rejects_invalid_delay_and_status() -> None:
    """Direct state use enforces the scenario bounds expected by test controls."""

    with pytest.raises(ValueError):
        ResponseScenario(delay_ms=30_001)
    with pytest.raises(ValueError):
        ResponseScenario(status_code=418)  # type: ignore[arg-type]
