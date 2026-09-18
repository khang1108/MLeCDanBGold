from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi import Response

from hcmai.api.result_logging import record_dres_result_log
from hcmai.vbs.models import ApiClientAnswer, RankedAnswer


class FakeVbsService:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.log_calls = []
        self.log_results = AsyncMock()
        if fail:
            self.log_results.side_effect = RuntimeError("DRES network error")

    def session_status(self, user_id: str):
        return {"user_id": user_id, "connected": True}

    async def resolve_evaluation(self, user_id: str):
        return "eval-1"

    def media_item_name(self, video_id: str) -> str:
        return f"mapped-{video_id}"


@pytest.mark.anyio
async def test_record_dres_result_log_preserves_order_and_ranks():
    vbs = FakeVbsService()
    service_container = {"vbs_service": vbs}
    response = Response()

    results = [
        SimpleNamespace(video_id="V2", timestamp_ms=2000),
        SimpleNamespace(video_id="V1", timestamp_ms=1000),
    ]

    await record_dres_result_log(
        service_container,
        response,
        user_id="team-a",
        category="TEXT",
        event_value="seafood",
        results=results,
    )

    assert response.headers["X-DRES-Log-Status"] == "sent"
    assert vbs.log_results.await_count == 1
    call_args = vbs.log_results.await_args.args
    assert call_args[0] == "team-a"
    assert call_args[1] == "eval-1"
    payload = call_args[2]

    assert len(payload.results) == 2
    assert payload.results[0].rank == 1
    assert payload.results[0].answer.media_item_name == "mapped-V2"
    assert payload.results[0].answer.start == 2000
    assert payload.results[1].rank == 2
    assert payload.results[1].answer.media_item_name == "mapped-V1"
    assert payload.results[1].answer.start == 1000


@pytest.mark.anyio
async def test_record_dres_result_log_handles_failure_gracefully():
    vbs = FakeVbsService(fail=True)
    service_container = {"vbs_service": vbs}
    response = Response()

    results = [SimpleNamespace(video_id="V1", timestamp_ms=1000)]

    # Must not raise exception
    await record_dres_result_log(
        service_container,
        response,
        user_id="team-a",
        category="TEXT",
        event_value="seafood",
        results=results,
    )

    assert response.headers["X-DRES-Log-Status"] == "failed"
