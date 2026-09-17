"""Session, evaluation-selection, and task-resolution tests for VBS service."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from hcmai.vbs.client import (
    DresAuthenticationError,
    DresNoActiveTaskError,
    DresUnavailableError,
)
from hcmai.vbs.config import DresCredential, DresSettings
from hcmai.vbs.models import (
    ApiClientSubmission,
    DresEvaluation,
    DresStatus,
    DresSubmissionStatus,
    DresTaskTemplateInfo,
    DresUser,
    QueryEvent,
    QueryResultLog,
)
from hcmai.vbs.service import (
    DresEvaluationSelectionError,
    DresService,
    DresUnknownUserError,
)
from pydantic import SecretStr


def _evaluation(identifier: str, status: str = "ACTIVE") -> DresEvaluation:
    """Build one official-shape active evaluation fixture."""

    return DresEvaluation.model_validate({
        "id": identifier,
        "name": "VBS test run",
        "type": "SYNCHRONOUS",
        "status": status,
        "templateId": "template-1",
        "teams": ["team"],
        "taskTemplates": [
            {
                "name": "KIS task",
                "taskGroup": "KIS",
                "taskType": "KIS",
                "duration": 300,
            }
        ],
    })


class _Client:
    """In-memory DRES client fake with call counts for session assertions."""

    def __init__(self) -> None:
        self.logins: list[tuple[str, str]] = []
        self.evaluations: list[DresEvaluation] = [_evaluation("eval-1")]
        self.task = DresTaskTemplateInfo.model_validate({
            "name": "KIS task",
            "taskGroup": "KIS",
            "taskType": "KIS",
            "duration": 300,
        })
        self.task_error: Exception | None = None
        self.fail_first_evaluation_call = False
        self.logged_results: list[tuple[str, str, object]] = []
        self.submit_errors: list[Exception] = []
        self.submissions: list[tuple[str, str, object]] = []
        self.state: dict[str, Any] = {"taskStatus": "RUNNING", "taskTemplateId": None}
        self.switches: list[int] = []
        self.starts: int = 0
        self.aborts: int = 0

    async def login(self, username: str, password: str) -> DresUser:
        self.logins.append((username, password))
        return DresUser.model_validate({
            "id": "dres-user",
            "username": username,
            "role": "PARTICIPANT",
            "sessionId": f"session-{len(self.logins)}",
        })

    async def list_evaluations(self, session_id: str) -> list[DresEvaluation]:
        if self.fail_first_evaluation_call and session_id == "session-1":
            raise DresAuthenticationError("expired")
        return self.evaluations

    async def get_current_task(self, evaluation_id: str, session_id: str) -> DresTaskTemplateInfo:
        if self.task_error is not None:
            raise self.task_error
        return self.task

    async def get_evaluation_state(self, evaluation_id: str, session_id: str) -> dict[str, Any]:
        return self.state

    async def switch_task(self, evaluation_id: str, task_idx: int, session_id: str) -> DresStatus:
        self.switches.append(task_idx)
        return DresStatus(status=True, description="switched")

    async def start_task(self, evaluation_id: str, session_id: str) -> DresStatus:
        self.starts += 1
        self.state["taskStatus"] = "RUNNING"
        return DresStatus(status=True, description="started")

    async def abort_task(self, evaluation_id: str, session_id: str) -> DresStatus:
        self.aborts += 1
        return DresStatus(status=True, description="aborted")

    async def log_results(self, evaluation_id: str, session_id: str, payload) -> None:
        """Capture actor session propagation for QueryResultLog calls."""

        self.logged_results.append((evaluation_id, session_id, payload))

    async def submit(self, evaluation_id: str, session_id: str, payload) -> DresSubmissionStatus:
        """Capture submit sessions and inject typed upstream failures."""

        self.submissions.append((evaluation_id, session_id, payload))
        if self.submit_errors:
            raise self.submit_errors.pop(0)
        return DresSubmissionStatus(
            status=True,
            submission="CORRECT",
            description="accepted",
        )

    async def aclose(self) -> None:
        return None


def _service(
    client: _Client,
    *,
    evaluation_id: str | None = None,
    with_admin: bool = False,
) -> DresService:
    """Create one service with two independent private participant mappings."""

    admin_credential = (
        DresCredential(username="admin", password=SecretStr("admin-pass"))
        if with_admin
        else None
    )
    settings = DresSettings(
        base_url="https://dres.test",
        evaluation_id=evaluation_id,
        credentials={
            "member-1": DresCredential(
                username="real-dres-user",
                password=SecretStr("private-password"),
            ),
            "member-2": DresCredential(
                username="other-dres-user",
                password=SecretStr("other-private-password"),
            ),
        },
        admin_credential=admin_credential,
    )
    return DresService(settings, client=client)


def test_connect_uses_server_credential_and_never_returns_dres_session() -> None:
    """Keep login credentials and the DRES session entirely backend-side."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client)

        result = await service.connect("member-1")
        again = await service.connect("member-1")

        assert result == {"user_id": "member-1", "connected": True}
        assert again == result
        assert client.logins == [("real-dres-user", "private-password")]
        assert "session-1" not in repr(result)
        assert "real-dres-user" not in repr(result)
        await service.aclose()

    asyncio.run(exercise())


def test_unknown_vbs_user_id_does_not_attempt_dres_login() -> None:
    """Explain a missing backend mapping without reaching DRES."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client)

        with pytest.raises(DresUnknownUserError, match="configured"):
            await service.connect("not-configured")

        assert client.logins == []
        await service.aclose()

    asyncio.run(exercise())


def test_concurrent_connects_for_one_user_share_one_login() -> None:
    """Serialize same-user handshakes while keeping other users independent."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client)
        first, second = await asyncio.gather(
            service.connect("member-1"),
            service.connect("member-1"),
        )

        assert first == second == {"user_id": "member-1", "connected": True}
        assert len(client.logins) == 1
        await service.aclose()

    asyncio.run(exercise())


def test_explicit_evaluation_id_skips_active_run_guessing() -> None:
    """Use configured evaluation directly without listing visible runs."""

    async def exercise() -> None:
        client = _Client()
        client.evaluations = []
        service = _service(client, evaluation_id="eval-explicit")
        await service.connect("member-1")

        assert await service.resolve_evaluation("member-1") == "eval-explicit"
        await service.aclose()

    asyncio.run(exercise())


def test_automatic_evaluation_selection_requires_exactly_one_active_run() -> None:
    """Never guess between zero or multiple visible ACTIVE evaluations."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client)
        await service.connect("member-1")
        assert await service.resolve_evaluation("member-1") == "eval-1"

        client.evaluations = [_evaluation("eval-ended", "TERMINATED")]
        with pytest.raises(DresEvaluationSelectionError, match="exactly one ACTIVE"):
            await service.resolve_evaluation("member-1")

        client.evaluations = [_evaluation("eval-1"), _evaluation("eval-2")]
        with pytest.raises(DresEvaluationSelectionError, match="exactly one ACTIVE"):
            await service.resolve_evaluation("member-1")
        await service.aclose()

    asyncio.run(exercise())


def test_dres_401_relogs_once_and_retries_with_new_private_session() -> None:
    """Refresh an expired DRES session once for the operation in progress."""

    async def exercise() -> None:
        client = _Client()
        client.fail_first_evaluation_call = True
        service = _service(client)
        await service.connect("member-1")

        result = await service.list_evaluations("member-1")

        assert [evaluation.id for evaluation in result] == ["eval-1"]
        assert client.logins == [
            ("real-dres-user", "private-password"),
            ("real-dres-user", "private-password"),
        ]
        await service.aclose()

    asyncio.run(exercise())


def test_submit_401_is_not_replayed_and_evicts_only_the_rejected_user() -> None:
    """Do not duplicate a POST; invalidate only the participant's old token."""

    async def exercise() -> None:
        client = _Client()
        client.submit_errors = [DresAuthenticationError("expired session")]
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")
        await service.connect("member-2")
        payload = ApiClientSubmission.model_validate({
            "answerSets": [{"taskName": "KIS task", "answers": [{"text": "answer"}]}],
        })

        with pytest.raises(DresAuthenticationError, match="expired session"):
            await service.submit("member-1", "eval-1", payload)

        assert [session for _evaluation, session, _payload in client.submissions] == ["session-1"]
        assert service.session_status("member-1")["connected"] is False
        assert service.session_status("member-2")["connected"] is True
        result = await service.submit("member-2", "eval-1", payload)
        assert result.status is True
        assert [session for _evaluation, session, _payload in client.submissions] == [
            "session-1",
            "session-2",
        ]
        assert len(client.logins) == 2
        await service.aclose()

    asyncio.run(exercise())


def test_submit_ambiguous_failure_does_not_login_or_retry() -> None:
    """Do not replay an answer after a timeout or other ambiguous transport failure."""

    async def exercise() -> None:
        client = _Client()
        client.submit_errors = [DresUnavailableError("network timeout")]
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")
        payload = ApiClientSubmission.model_validate({
            "answerSets": [{"taskName": "KIS task", "answers": [{"text": "answer"}]}],
        })

        with pytest.raises(DresUnavailableError, match="timeout"):
            await service.submit("member-1", "eval-1", payload)

        assert [session for _evaluation, session, _payload in client.submissions] == ["session-1"]
        assert len(client.logins) == 1
        await service.aclose()

    asyncio.run(exercise())


def test_result_log_uses_the_session_of_the_participant_who_searched() -> None:
    """Never substitute a shared/team session when forwarding a search log."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")
        payload = QueryResultLog(
            timestamp=1_800_000_000_000,
            sort_type="list",
            result_set_availability="",
            results=[],
            events=[QueryEvent(
                timestamp=1_800_000_000_000,
                category="TEXT",
                event_type="SEARCH",
                value="a person running",
            )],
        )

        await service.log_results("member-1", "eval-1", payload)

        assert client.logged_results == [("eval-1", "session-1", payload)]
        await service.aclose()

    asyncio.run(exercise())


def test_current_task_accepts_official_template_info_without_a_task_id() -> None:
    """Use the official current-task fields directly without inventing an ID."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")

        task = await service.resolve_task("member-1", "eval-1")

        assert task.name == "KIS task"
        assert task.task_group == "KIS"
        assert task.task_type == "KIS"
        assert not hasattr(task, "task_id")
        await service.aclose()

    asyncio.run(exercise())


def test_task_scope_key_is_stable_and_changes_with_task_or_evaluation() -> None:
    """Derive deterministic, backend-only identity from exact current-task data."""

    async def exercise() -> None:
        client = _Client()
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")

        first = await service.resolve_scope("member-1")
        repeated = await service.resolve_scope("member-1")
        canonical = json.dumps(
            {
                "evaluation_id": "eval-1",
                "name": "KIS task",
                "taskGroup": "KIS",
                "taskType": "KIS",
                "duration": 300,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_key = "dres-task-v1:" + hashlib.sha256(canonical).hexdigest()

        assert first == repeated
        assert first.evaluation_id == "eval-1"
        assert first.task_scope_key == expected_key
        assert first.task_name == "KIS task"

        client.task = DresTaskTemplateInfo.model_validate({
            "name": "VQA task",
            "taskGroup": "VQA",
            "taskType": "VQA",
            "duration": 300,
        })
        changed_task = await service.resolve_scope("member-1")
        assert changed_task.task_scope_key != first.task_scope_key

        other_evaluation_service = _service(client, evaluation_id="eval-2")
        await other_evaluation_service.connect("member-1")
        changed_evaluation = await other_evaluation_service.resolve_scope("member-1")
        assert changed_evaluation.task_scope_key != changed_task.task_scope_key

        await service.aclose()
        await other_evaluation_service.aclose()

    asyncio.run(exercise())


def test_no_active_current_task_remains_a_typed_dres_error() -> None:
    """Propagate DRES's not-found result without fabricating task metadata."""

    async def exercise() -> None:
        client = _Client()
        client.task_error = DresNoActiveTaskError("no current task")
        service = _service(client, evaluation_id="eval-1")
        await service.connect("member-1")

        with pytest.raises(DresNoActiveTaskError, match="no current task"):
            await service.resolve_scope("member-1")

        await service.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "video_id,expected",
    [
        ("00001", "00001"),
        ("prefix.00002", "00002"),
        ("other.prefix.00003", "other.prefix.00003"),
    ],
)
def test_media_item_mapping_is_identity_or_exact_anchored_prefix_strip(
    video_id: str,
    expected: str,
) -> None:
    """Never apply display-ID heuristics or strip a non-leading prefix."""

    client = _Client()
    service = _service(client)
    service.settings = DresSettings(
        base_url="https://dres.test",
        media_id_prefix_to_strip="prefix.",
        credentials=service.settings.credentials,
    )

    assert service.media_item_name(video_id) == expected


@pytest.mark.parametrize("timestamp_ms", [-1, 1.5, True, "1000"])
def test_temporal_mapping_rejects_non_integer_or_negative_time(timestamp_ms) -> None:
    """Accept only canonical non-negative integer media milliseconds."""

    service = _service(_Client())

    with pytest.raises(ValueError, match="timestamp_ms"):
        service.temporal_answer("video-a", timestamp_ms)


def test_temporal_answer_preserves_exact_milliseconds_and_never_uses_frame_index() -> None:
    """Use only video ID and selected timestamp for DRES temporal coordinates."""

    service = _service(_Client())
    answer = service.temporal_answer("video.with.dots", 12_346)

    assert answer.model_dump(by_alias=True, exclude_none=True) == {
        "mediaItemName": "video.with.dots",
        "start": 12_346,
        "end": 12_346,
    }


def test_temporal_range_answer_preserves_both_exact_interval_bounds() -> None:
    """Keep the submitted event interval instead of collapsing it to a point."""

    service = _service(_Client())

    answer = service.temporal_range_answer("video.with.dots", 1_234, 5_678)

    assert answer.model_dump(by_alias=True, exclude_none=True) == {
        "mediaItemName": "video.with.dots",
        "start": 1_234,
        "end": 5_678,
    }


@pytest.mark.parametrize(
    "start_ms,end_ms",
    [(-1, 0), (0, -1), (True, 1), (1, 2.5), (2, 1)],
)
def test_temporal_range_answer_rejects_invalid_interval(start_ms, end_ms) -> None:
    """Reject non-integer, negative, and reversed milliseconds before posting."""

    service = _service(_Client())

    with pytest.raises(ValueError):
        service.temporal_range_answer("video-a", start_ms, end_ms)


@pytest.mark.parametrize("video_id", ["", "  ", "."])
def test_temporal_mapping_rejects_blank_or_prefix_only_media_ids(video_id: str) -> None:
    """Fail closed if canonical identity maps to an empty DRES media name."""

    service = _service(_Client())
    if video_id == ".":
        service.settings = DresSettings(
            base_url="https://dres.test",
            media_id_prefix_to_strip=".",
            credentials=service.settings.credentials,
        )

    with pytest.raises(ValueError, match="media item name"):
        service.temporal_answer(video_id, 0)


def test_service_ensure_task_running_aborts_switches_and_starts() -> None:
    """If admin is configured and task is not running, ensure_task_running activates it."""

    async def exercise() -> None:
        client = _Client()
        client.state = {"taskStatus": "ENDED", "taskTemplateId": None}
        service = _service(client, with_admin=True)

        ok = await service.ensure_task_running("eval-1", "KIS task")
        assert ok is True
        assert len(client.switches) == 1
        assert client.switches[0] == 0
        assert client.starts == 1

    asyncio.run(exercise())


def test_service_submit_auto_activates_and_retries_when_task_not_running() -> None:
    """When submission fails because task is not running, activate task and retry once."""

    from hcmai.vbs.client import DresRejectedSubmissionError

    async def exercise() -> None:
        client = _Client()
        client.state = {"taskStatus": "ENDED", "taskTemplateId": None}
        # First submit call fails with 'not running', retry succeeds
        client.submit_errors = [
            DresRejectedSubmissionError("Task run is currently not running.")
        ]
        service = _service(client, with_admin=True)
        await service.connect("member-1")

        payload = ApiClientSubmission(answer_sets=[])
        res = await service.submit("member-1", "eval-1", payload, task_name="KIS task")
        assert res.submission == "CORRECT"
        assert len(client.submissions) == 2
        assert client.starts == 1

    asyncio.run(exercise())

