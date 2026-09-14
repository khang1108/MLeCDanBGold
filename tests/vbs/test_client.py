"""Mocked HTTP tests for exact DRES v2 paths, payloads, and failures."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from hcmai.vbs.client import (
    DresAuthenticationError,
    DresClient,
    DresNoActiveTaskError,
    DresRejectedSubmissionError,
    DresUnavailableError,
)
from hcmai.vbs.config import DresSettings
from hcmai.vbs.models import (
    ApiClientAnswer,
    ApiClientAnswerSet,
    ApiClientSubmission,
    DresSubmissionStatus,
    QueryEvent,
    QueryResultLog,
)


def _settings() -> DresSettings:
    """Return deterministic client configuration without participant secrets."""

    return DresSettings.from_env({"HCMAI_DRES_BASE_URL": "https://dres.test"})


def test_client_uses_exact_dres_v2_paths_query_and_body_casing() -> None:
    """Exercise the complete client surface with exact transport assertions."""

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v2/login":
            assert request.read() == b'{"username":"team","password":"pw"}'
            return httpx.Response(200, json={
                "id": "member-1",
                "username": "team",
                "role": "PARTICIPANT",
                "sessionId": "secret-session",
            })
        if request.url.path == "/api/v2/client/evaluation/list":
            return httpx.Response(200, json=[{
                "id": "eval-1",
                "name": "Test run",
                "type": "SYNCHRONOUS",
                "status": "ACTIVE",
                "templateId": "template-1",
                "teams": ["team"],
                "taskTemplates": [],
            }])
        if request.url.path == "/api/v2/client/evaluation/currentTask/eval-1":
            return httpx.Response(200, json={
                "name": "KIS",
                "taskGroup": "KIS",
                "taskType": "KIS",
            })
        if request.url.path == "/api/v2/submit/eval-1":
            assert request.read() == (
                b'{"answerSets":[{"taskName":"KIS","answers":'
                b'[{"mediaItemName":"video-1","start":12345,"end":12345}]}]}'
            )
            return httpx.Response(202, json={
                "status": True,
                "submission": "INDETERMINATE",
                "description": "accepted",
            })
        if request.url.path == "/api/v2/log/result/eval-1":
            assert request.read() == (
                b'{"timestamp":1800000000000,"sortType":"list",'
                b'"resultSetAvailability":"","results":[{"answer":'
                b'{"mediaItemName":"video-1","start":12345,"end":12345},'
                b'"rank":1}],"events":'
                b'[{"timestamp":1800000000000,"category":"TEXT",'
                b'"type":"SEARCH","value":"person running"}]}'
            )
            return httpx.Response(200, json={"status": True, "description": "logged"})
        raise AssertionError(f"Unexpected DRES request {request.method} {request.url.path}")

    async def exercise() -> None:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))
        user = await client.login("team", "pw")
        evaluations = await client.list_evaluations(user.session_id)
        task = await client.get_current_task(evaluations[0].id, user.session_id)
        submission = ApiClientSubmission(answer_sets=[ApiClientAnswerSet(
            task_name=task.name,
            answers=[ApiClientAnswer(media_item_name="video-1", start=12345, end=12345)],
        )])
        result = await client.submit(evaluations[0].id, user.session_id, submission)
        assert result.submission == "INDETERMINATE"
        log = QueryResultLog(
            timestamp=1_800_000_000_000,
            sort_type="list",
            result_set_availability="",
            results=[{
                "rank": 1,
                "answer": {
                    "mediaItemName": "video-1",
                    "start": 12345,
                    "end": 12345,
                },
            }],
            events=[QueryEvent(
                timestamp=1_800_000_000_000,
                category="TEXT",
                event_type="SEARCH",
                value="person running",
            )],
        )
        await client.log_results(evaluations[0].id, user.session_id, log)
        await client.aclose()

    asyncio.run(exercise())

    assert [request.url.path for request in requests] == [
        "/api/v2/login",
        "/api/v2/client/evaluation/list",
        "/api/v2/client/evaluation/currentTask/eval-1",
        "/api/v2/submit/eval-1",
        "/api/v2/log/result/eval-1",
    ]
    assert requests[1].url.params["session"] == "secret-session"
    assert requests[2].url.params["session"] == "secret-session"
    assert requests[3].url.params["session"] == "secret-session"
    assert requests[4].url.params["session"] == "secret-session"
    legacy_query_log_path = "/api/v2/log/" + "query"
    assert not any(legacy_query_log_path in request.url.path for request in requests)


@pytest.mark.parametrize(
    "status_code,verdict",
    [(200, "CORRECT"), (202, "INDETERMINATE")],
)
def test_submit_parses_official_success_status(
    status_code: int,
    verdict: str,
) -> None:
    """Parse both verdict-bearing DRES success status codes exactly."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v2/submit/eval-1"
        assert request.url.params["session"] == "private-session"
        assert request.read() == (
            b'{"answerSets":[{"taskName":"KIS task","answers":'
            b'[{"mediaItemName":"video-1","start":10,"end":10}]}]}'
        )
        return httpx.Response(status_code, json={
            "status": True,
            "submission": verdict,
            "description": "accepted",
        })

    async def exercise() -> DresSubmissionStatus:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))
        payload = ApiClientSubmission(answer_sets=[ApiClientAnswerSet(
            task_name="KIS task",
            answers=[ApiClientAnswer(media_item_name="video-1", start=10, end=10)],
        )])
        result = await client.submit("eval-1", "private-session", payload)
        await client.aclose()
        return result

    result = asyncio.run(exercise())

    assert result.status is True
    assert result.submission == verdict
    assert result.description == "accepted"


@pytest.mark.parametrize(
    "status_code,body",
    [
        (200, {"status": True, "description": "missing verdict"}),
        (200, {"status": True, "submission": "MAYBE", "description": "unknown"}),
        (202, None),
    ],
)
def test_submit_rejects_malformed_or_empty_success_bodies(
    status_code: int,
    body: dict[str, object] | None,
) -> None:
    """Do not claim delivery when DRES omitted or corrupted its verdict."""

    def handler(_: httpx.Request) -> httpx.Response:
        if body is None:
            return httpx.Response(status_code, content=b"")
        return httpx.Response(status_code, json=body)

    async def exercise() -> None:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))
        payload = ApiClientSubmission(answer_sets=[ApiClientAnswerSet(
            task_name="KIS task",
            answers=[ApiClientAnswer(media_item_name="video-1", start=10, end=10)],
        )])
        try:
            with pytest.raises(DresUnavailableError):
                await client.submit("eval-1", "private-session", payload)
        finally:
            await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "status_code,exception",
    [
        (400, DresRejectedSubmissionError),
        (401, DresAuthenticationError),
        (404, DresNoActiveTaskError),
        (412, DresRejectedSubmissionError),
    ],
)
def test_client_translates_dres_error_statuses_without_session_leak(
    status_code: int,
    exception: type[Exception],
) -> None:
    """Classify expected DRES errors and redact query-string session tokens."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"status": False, "description": "bad session=secret-session"},
        )

    async def exercise() -> None:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))
        submission = ApiClientSubmission(answer_sets=[ApiClientAnswerSet(
            task_name="KIS task",
            answers=[ApiClientAnswer(media_item_name="video-1", start=1, end=1)],
        )])

        with pytest.raises(exception) as error:
            if exception is DresNoActiveTaskError:
                await client.get_current_task("eval-1", "secret-session")
            else:
                await client.submit("eval-1", "secret-session", submission)

        assert "secret-session" not in str(error.value)
        assert "session=secret-session" not in str(error.value)
        await client.aclose()

    asyncio.run(exercise())


def test_client_redacts_passwords_from_authentication_errors() -> None:
    """Do not trust upstream descriptions to omit submitted credentials."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"status": False, "description": "invalid password: super-secret"},
        )

    async def exercise() -> None:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))

        with pytest.raises(DresAuthenticationError) as error:
            await client.login("team", "super-secret")

        assert "super-secret" not in str(error.value)
        await client.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("late")])
def test_client_translates_network_failures_without_request_url(failure: Exception) -> None:
    """Avoid leaking session query tokens in httpx exception URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    async def exercise() -> None:
        client = DresClient(_settings(), transport=httpx.MockTransport(handler))

        with pytest.raises(DresUnavailableError) as error:
            await client.list_evaluations("secret-session")

        assert "secret-session" not in str(error.value)
        await client.aclose()

    asyncio.run(exercise())
