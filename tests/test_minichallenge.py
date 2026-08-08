"""Contract, adapter, and API tests for the optional DRES integration."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import cast

import httpx
import pytest

from hcmai.app import create_app
from hcmai.common.schemas import FrameRecord, MiniChallengeSubmitRequest
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retriever.pipeline import RetrievalService
from hcmai.submission.adapters import DRESClient, DRESClientError
from hcmai.submission.pipeline import MiniChallengeService


EVALUATION = {
    "id": "evaluation-1",
    "name": "Mini QA",
    "type": "SYNCHRONOUS",
    "status": "ACTIVE",
    "templateId": "template-1",
    "teams": ["team-1"],
    "taskTemplates": [
        {"name": "QA task", "taskGroup": "qa", "taskType": "QA"}
    ],
}
TASK = {"name": "QA task", "taskGroup": "qa", "taskType": "QA"}
RESULT = {
    "status": True,
    "submission": "INDETERMINATE",
    "description": "Submission accepted",
}


class FrameStore:
    record = FrameRecord(
        frame_id="frame-90",
        video_id="L21_V001",
        frame_idx=90,
        timestamp_ms=3_600,
        image_path="frames/90.jpg",
        width=1,
        height=1,
    )

    def get(self, frame_id: str) -> FrameRecord:
        if frame_id != self.record.frame_id:
            raise KeyError(frame_id)
        return self.record

    get_frame = get


def _services(handler):
    http_client = httpx.AsyncClient(
        base_url="http://dres.test/",
        transport=httpx.MockTransport(handler),
    )
    mini = MiniChallengeService(DRESClient("http://dres.test", client=http_client))
    search = SearchService(
        cast(DataService, FrameStore()), cast(RetrievalService, object())
    )
    return search, mini, http_client


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://local.test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_dres_flow_preserves_session_and_exact_submission_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/evaluation/list"):
            return httpx.Response(200, json=[EVALUATION])
        if "/currentTask/" in request.url.path:
            return httpx.Response(200, json=TASK)
        return httpx.Response(202, json=RESULT)

    search, mini, client = _services(handler)

    async def run() -> None:
        evaluations = await mini.list_evaluations("private-session")
        task = await mini.current_task(evaluations[0].id, "private-session")
        result = await mini.submit_frame(
            evaluations[0].id,
            "private-session",
            MiniChallengeSubmitRequest(
                frame_id="frame-90", task_name=task.name, text="Bơ"
            ),
            search.get_frame("frame-90"),
        )
        assert result.submission == "INDETERMINATE"
        await client.aclose()

    asyncio.run(run())
    assert [request.url.params["session"] for request in requests] == [
        "private-session"
    ] * 3
    assert json.loads(requests[-1].content) == {
        "answerSets": [{
            "taskName": "QA task",
            "answers": [{
                "mediaItemName": "L21_V001",
                "start": 3_600,
                "end": 3_600,
                "text": "Bơ",
            }],
        }]
    }


def test_minichallenge_api_requires_header_and_resolves_frame_identity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/login"):
            return httpx.Response(
                200,
                json={
                    "id": "user-1",
                    "username": "participant-1",
                    "role": "PARTICIPANT",
                    "sessionId": "session-1",
                },
            )
        if request.method == "POST":
            return httpx.Response(200, json={**RESULT, "submission": "CORRECT"})
        return httpx.Response(200, json=[EVALUATION])

    search, mini, client = _services(handler)
    app = create_app(search, mini)
    login_resp = _request(
        app,
        "POST",
        "/api/v1/minichallenge/login",
        json={"username": "participant-1", "password": "secret-password"},
    )
    missing = _request(app, "GET", "/api/v1/minichallenge/evaluations")
    listed = _request(
        app,
        "GET",
        "/api/v1/minichallenge/evaluations",
        headers={"X-DRES-Session": "session-1"},
    )
    submitted = _request(
        app,
        "POST",
        "/api/v1/minichallenge/evaluations/evaluation-1/submit",
        headers={"X-DRES-Session": "session-1"},
        json={"frame_id": "frame-90", "task_name": "QA task", "text": "Bơ"},
    )
    unknown = _request(
        app,
        "POST",
        "/api/v1/minichallenge/evaluations/evaluation-1/submit",
        headers={"X-DRES-Session": "session-1"},
        json={"frame_id": "unknown", "task_name": "QA task"},
    )
    asyncio.run(client.aclose())

    assert login_resp.status_code == 200
    assert login_resp.json()["sessionId"] == "session-1"
    assert missing.status_code == 422
    assert listed.json()[0]["taskTemplates"][0]["taskType"] == "QA"
    assert submitted.json()["submission"] == "CORRECT"
    assert unknown.status_code == 404


@pytest.mark.parametrize(
    ("raised", "status_code", "message"),
    [
        (httpx.ReadTimeout("slow"), 504, "timed out"),
        (httpx.ConnectError("offline"), 502, "Could not reach"),
    ],
)
def test_dres_transport_failures_are_bounded(
    raised: Exception, status_code: int, message: str
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise raised

    _, mini, client = _services(handler)
    with pytest.raises(DRESClientError, match=message) as error:
        asyncio.run(mini.list_evaluations("session"))
    asyncio.run(client.aclose())
    assert error.value.status_code == status_code


def test_dres_rejection_preserves_safe_status_without_retry() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            412, json={"status": False, "description": "Task is closed"}
        )

    _, mini, client = _services(handler)
    with pytest.raises(DRESClientError, match="Task is closed") as error:
        asyncio.run(mini.list_evaluations("session"))
    asyncio.run(client.aclose())
    assert error.value.status_code == 412
    assert calls == 1


def test_dres_session_is_absent_from_logs(caplog: pytest.LogCaptureFixture) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[EVALUATION])

    _, mini, client = _services(handler)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(mini.list_evaluations("never-log-this-session"))
    asyncio.run(client.aclose())

    output = "\n".join(record.getMessage() for record in caplog.records)
    assert "DRES request completed" in output
    assert "never-log-this-session" not in output
