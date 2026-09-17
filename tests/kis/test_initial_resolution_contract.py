import pytest
from pydantic import ValidationError

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent


def test_initial_resolution_schema_contains_only_events() -> None:
    schema = KISInitialResolution.model_json_schema()
    assert set(schema["properties"]) == {"events"}


def test_initial_event_rejects_text_over_240_characters() -> None:
    with pytest.raises(ValidationError):
        KISInitialResolutionEvent(text="x" * 241)


def test_initial_resolution_rejects_too_many_events() -> None:
    events = [
        {"text": f"Distinct visual moment {index}."}
        for index in range(DEFAULT_MAX_TEMPORAL_EVENT_COUNT + 1)
    ]
    with pytest.raises(ValidationError):
        KISInitialResolution(events=events)
