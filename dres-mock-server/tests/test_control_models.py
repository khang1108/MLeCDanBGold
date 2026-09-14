"""Contract tests for strict test-control request models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dres_mock_server.models import (
    DresVerdict,
    ResultLogScenarioRequest,
    SubmissionScenarioRequest,
    TaskControlRequest,
)


def test_task_control_aliases_and_exact_nonblank_values_are_preserved() -> None:
    """Task metadata uses DRES casing and keeps meaningful surrounding spaces."""

    request = TaskControlRequest.model_validate(
        {
            "active": True,
            "name": " AVS task ",
            "taskGroup": " AVS ",
            "taskType": " AVS ",
            "duration": 300,
        }
    )

    assert request.model_dump(by_alias=True) == {
        "active": True,
        "name": " AVS task ",
        "taskGroup": " AVS ",
        "taskType": " AVS ",
        "duration": 300,
    }


@pytest.mark.parametrize("field", ["name", "taskGroup", "taskType"])
@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_task_control_rejects_blank_metadata(field: str, value: str) -> None:
    """Task labels must contain non-whitespace text."""

    payload = {
        "active": True,
        "name": "AVS task",
        "taskGroup": "AVS",
        "taskType": "AVS",
        "duration": 300,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        TaskControlRequest.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "active": True,
            "name": "AVS task",
            "taskGroup": "AVS",
            "taskType": "AVS",
            "duration": 300,
            "unexpected": "field",
        },
        {
            "active": 1,
            "name": "AVS task",
            "taskGroup": "AVS",
            "taskType": "AVS",
            "duration": 300,
        },
    ],
)
def test_task_control_rejects_extra_fields_and_coercion(payload: dict[str, object]) -> None:
    """Control payloads are closed and use strict primitive types."""

    with pytest.raises(ValidationError):
        TaskControlRequest.model_validate(payload)


@pytest.mark.parametrize("delay_ms", [0, 30_000])
def test_submission_scenario_accepts_bounded_delay_and_serializes_aliases(
    delay_ms: int,
) -> None:
    """Submission scenario aliases serialize to the control API wire shape."""

    request = SubmissionScenarioRequest.model_validate(
        {
            "statusCode": 202,
            "verdict": "INDETERMINATE",
            "delayMs": delay_ms,
            "malformedBody": False,
            "description": "queued for judging",
        }
    )

    assert request.status_code == 202
    assert request.verdict is DresVerdict.INDETERMINATE
    assert request.model_dump(by_alias=True) == {
        "statusCode": 202,
        "verdict": "INDETERMINATE",
        "delayMs": delay_ms,
        "malformedBody": False,
        "description": "queued for judging",
    }


@pytest.mark.parametrize("delay_ms", [-1, 30_001, 1.5, True])
def test_submission_scenario_rejects_out_of_range_or_non_integer_delay(
    delay_ms: object,
) -> None:
    """Delay must be a strict integer inside the supported 30-second bound."""

    with pytest.raises(ValidationError):
        SubmissionScenarioRequest.model_validate({"delayMs": delay_ms})


def test_scenario_models_forbid_unknown_fields_and_invalid_status_or_verdict() -> None:
    """Only explicitly supported response controls may be configured."""

    invalid_payloads = [
        (SubmissionScenarioRequest, {"statusCode": 201}),
        (SubmissionScenarioRequest, {"verdict": "MAYBE"}),
        (SubmissionScenarioRequest, {"unknown": "value"}),
        (ResultLogScenarioRequest, {"statusCode": 201}),
        (ResultLogScenarioRequest, {"verdict": "WRONG"}),
        (ResultLogScenarioRequest, {"extra": True}),
    ]

    for model, payload in invalid_payloads:
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_result_log_scenario_uses_supported_fields_and_defaults() -> None:
    """Result-log controls omit submission-only verdicts and use safe defaults."""

    request = ResultLogScenarioRequest()

    assert request.model_dump(by_alias=True) == {
        "statusCode": 200,
        "delayMs": 0,
        "malformedBody": False,
        "description": "mock accepted",
    }
