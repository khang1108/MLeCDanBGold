"""Contract tests for DRES result-log payload models."""

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from dres_mock_server.models import QueryResultLog


def test_result_log_preserves_aliases_values_and_order() -> None:
    """Required result fields retain timestamps, event order, and media data."""

    log = QueryResultLog(
        timestamp=1_800_000_000_000,
        sortType="list",
        resultSetAvailability="",
        results=[
            {"rank": 1, "answer": {"mediaItemName": "00001", "start": 1234, "end": 1234}},
            {"rank": 2, "answer": {"mediaItemName": "00002", "start": 5678, "end": 6789}},
        ],
        events=[
            {
                "timestamp": 1_800_000_000_000,
                "category": "TEXT",
                "type": "SEARCH",
                "value": "person running",
            }
        ],
    )

    assert log.model_dump(by_alias=True, exclude_unset=True) == {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 1, "answer": {"mediaItemName": "00001", "start": 1234, "end": 1234}},
            {"rank": 2, "answer": {"mediaItemName": "00002", "start": 5678, "end": 6789}},
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


def test_result_log_arrays_may_be_empty_and_rank_may_be_omitted() -> None:
    """The OpenAPI requires array fields, not elements or ranked positions."""

    empty_log = QueryResultLog(
        timestamp=1_800_000_000_000,
        sortType="list",
        resultSetAvailability="",
        results=[],
        events=[],
    )
    unranked_log = QueryResultLog(
        timestamp=1_800_000_000_000,
        sortType="list",
        resultSetAvailability="",
        results=[{"answer": {"text": "three"}}],
        events=[],
    )

    assert empty_log.results == []
    assert empty_log.events == []
    assert unranked_log.results[0].rank is None
    assert unranked_log.model_dump(by_alias=True, exclude_none=True)["results"] == [
        {"answer": {"text": "three"}}
    ]


@pytest.mark.parametrize(
    "missing_field",
    ["timestamp", "sortType", "resultSetAvailability", "results", "events"],
)
def test_result_log_requires_every_official_field(missing_field: str) -> None:
    """Empty arrays and availability text remain required properties."""

    payload: dict[str, object] = {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [],
        "events": [],
    }
    del payload[missing_field]

    with pytest.raises(ValidationError):
        QueryResultLog.model_validate(payload)


def test_result_log_rejects_unknown_fields_and_non_integer_timestamps() -> None:
    """Strict wire validation prevents extensions and numeric coercion."""

    with pytest.raises(ValidationError):
        QueryResultLog.model_validate(
            {
                "timestamp": 1_800_000_000_000,
                "sortType": "list",
                "resultSetAvailability": "",
                "results": [],
                "events": [],
                "debug": True,
            }
        )

    with pytest.raises(ValidationError):
        QueryResultLog(
            timestamp="1800000000000",
            sortType="list",
            resultSetAvailability="",
            results=[],
            events=[],
        )


def test_result_log_endpoint_captures_exact_payload_and_returns_only_status(
    client: TestClient,
    session_id: str,
) -> None:
    """The result-log endpoint preserves events/results without claiming a score."""

    payload = {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [
            {"rank": 1, "answer": {"mediaItemName": "00001", "start": 1234, "end": 1234}},
            {"rank": 2, "answer": {"mediaItemName": "00002", "start": 5678, "end": 5678}},
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

    response = client.post(
        "/api/v2/log/result/eval-vbs-local",
        params={"session": session_id},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json() == {"status": True, "description": "mock accepted"}
    record = client.app.state.mock_state.captured_requests("RESULT_LOG")[0]
    assert record.payload == payload
    assert [item["rank"] for item in record.payload["results"]] == [1, 2]
    assert record.payload["resultSetAvailability"] == ""
    assert record.outcome.verdict is None


def test_result_log_endpoint_rejects_invalid_session_and_evaluation(
    client: TestClient,
) -> None:
    """Untrusted or wrong-evaluation logs are not added to captured traffic."""

    payload = {
        "timestamp": 1_800_000_000_000,
        "sortType": "list",
        "resultSetAvailability": "",
        "results": [],
        "events": [],
    }
    invalid_session = client.post(
        "/api/v2/log/result/eval-vbs-local",
        params={"session": "expired-session"},
        json=payload,
    )
    session = client.post(
        "/api/v2/login",
        json={"username": "member-1", "password": "password-1"},
    ).json()["sessionId"]
    wrong_evaluation = client.post(
        "/api/v2/log/result/other-evaluation",
        params={"session": session},
        json=payload,
    )

    assert invalid_session.status_code == 401
    assert wrong_evaluation.status_code == 404
    assert client.app.state.mock_state.captured_requests("RESULT_LOG") == []


def test_result_log_validation_failure_returns_400_without_capture(
    client: TestClient,
    session_id: str,
) -> None:
    """Missing official fields are reported with the DRES HTTP 400 contract."""

    response = client.post(
        "/api/v2/log/result/eval-vbs-local",
        params={"session": session_id},
        json={"timestamp": 1_800_000_000_000, "results": [], "events": []},
    )

    assert response.status_code == 400
    assert response.json()["status"] is False
    assert client.app.state.mock_state.captured_requests("RESULT_LOG") == []
