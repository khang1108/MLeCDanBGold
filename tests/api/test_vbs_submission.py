from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hcmai.api.routers.vbs import create_vbs_router
from hcmai.vbs.models import ApiClientAnswer, DresSubmissionStatus


class FakeDresService:
    def __init__(self, *, task_group: str, task_type: str, scope_key: str = "scope-1") -> None:
        self.scope = SimpleNamespace(
            evaluation_id="eval-1",
            task_scope_key=scope_key,
            task_name="Task 1",
            task_group=task_group,
            task_type=task_type,
            duration=300,
        )
        self.submit = AsyncMock(
            return_value=DresSubmissionStatus(
                status=True,
                submission="CORRECT",
                description="recorded",
            )
        )

    def session_status(self, user_id: str) -> dict[str, object]:
        return {"user_id": user_id, "connected": True}

    async def resolve_scope(self, user_id: str, *, evaluation_id=None, task_name=None):
        return self.scope

    def temporal_range_answer(self, video_id: str, start_ms: int, end_ms: int) -> ApiClientAnswer:
        return ApiClientAnswer(
            media_item_name=video_id,
            start=start_ms,
            end=end_ms,
        )


def make_client(*, task_group: str, task_type: str, scope_key: str = "scope-1"):
    service = FakeDresService(
        task_group=task_group,
        task_type=task_type,
        scope_key=scope_key,
    )
    app = FastAPI()
    app.include_router(create_vbs_router({"vbs_service": service}))
    return TestClient(app), service


def temporal(video_id: str, timestamp_ms: int) -> dict[str, object]:
    return {
        "kind": "TEMPORAL",
        "video_id": video_id,
        "start_ms": timestamp_ms,
        "end_ms": timestamp_ms,
    }


def submit_body(answers: list[dict[str, object]], *, scope_key: str = "scope-1") -> dict[str, object]:
    return {
        "user_id": "team-a",
        "expected_task_scope_key": scope_key,
        "answers": answers,
    }


def test_kis_accepts_exactly_one_temporal_answer():
    client, service = make_client(task_group="KIS", task_type="KNOWN ITEM SEARCH")

    response = client.post("/api/v1/vbs/submit", json=submit_body([temporal("V1", 1000)]))

    assert response.status_code == 200
    assert response.json()["state"] == "RECORDED"
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert len(payload.answer_sets) == 1
    assert len(payload.answer_sets[0].answers) == 1


def test_kis_rejects_multiple_answers_before_dres_call():
    client, service = make_client(task_group="KIS", task_type="KNOWN ITEM SEARCH")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([temporal("V1", 1000), temporal("V2", 2000)]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_avs_accepts_multiple_temporal_answers_in_one_answer_set():
    client, service = make_client(task_group="AVS", task_type="AD-HOC VIDEO SEARCH")
    answers = [temporal(f"V{i}", i * 1000) for i in range(1, 21)]

    response = client.post("/api/v1/vbs/submit", json=submit_body(answers))

    assert response.status_code == 200
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert len(payload.answer_sets) == 1
    assert payload.answer_sets[0].task_name == "Task 1"
    assert len(payload.answer_sets[0].answers) == 20


def test_avs_rejects_text_answer_before_dres_call():
    client, service = make_client(task_group="AVS", task_type="AVS")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([{"kind": "TEXT", "text": "not temporal"}]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_vqa_accepts_exactly_one_text_answer():
    client, service = make_client(task_group="QA", task_type="VQA")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([{"kind": "TEXT", "text": "blue"}]),
    )

    assert response.status_code == 200
    assert service.submit.await_count == 1
    payload = service.submit.await_args.args[2]
    assert payload.answer_sets[0].answers[0].text == "blue"


def test_vqa_rejects_multiple_answers_before_dres_call():
    client, service = make_client(task_group="QA", task_type="VQA")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([
            {"kind": "TEXT", "text": "blue"},
            {"kind": "TEXT", "text": "green"},
        ]),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANSWER_KIND_MISMATCH"
    service.submit.assert_not_awaited()


def test_stale_scope_is_rejected_before_dres_call():
    client, service = make_client(task_group="AVS", task_type="AVS", scope_key="live-scope")

    response = client.post(
        "/api/v1/vbs/submit",
        json=submit_body([temporal("V1", 1000)], scope_key="stale-scope"),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "TASK_SCOPE_MISMATCH"
    service.submit.assert_not_awaited()
