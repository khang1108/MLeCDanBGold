"""Focused contract and regression tests for KIS initial temporal decomposition.

This test module verifies the prompt and structured-output schema contracts that
prevent the initial resolver from collapsing multi-moment natural language narratives
into a single event object.
"""

from typing import Any

from hcmai.kis.models import KISInitialResolution, KISInitialResolutionEvent
from hcmai.kis.resolution import (
    KIS_INITIAL_RESOLVER_SYSTEM_PROMPT,
    KISIntentResolver,
    build_kis_initial_messages,
)


def test_initial_prompt_states_one_clue_can_produce_multiple_events() -> None:
    messages = build_kis_initial_messages(
        ["A person enters a room. Then the person sits down."]
    )

    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "One event is one distinct retrievable visual moment" in system
    assert "A single clue may contain multiple events" in user


def test_initial_prompt_requires_sequential_moments_to_be_separate_events() -> None:
    system = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT

    assert "Sequential views or actions are separate events" in system
    assert "Do not explain reasoning" in system


def test_initial_prompt_contains_non_cycling_one_clue_multi_event_example() -> None:
    system = KIS_INITIAL_RESOLVER_SYSTEM_PROMPT

    assert "The camera starts on a framed certificate" in system
    assert "Events:" in system
    assert "1. A framed certificate" in system
    assert "2. A craftsperson engraves" in system


def test_resolution_schema_describes_events_as_distinct_temporal_moments() -> None:
    schema = KISInitialResolution.model_json_schema()
    events = schema["properties"]["events"]

    description = events["description"]
    assert "Chronologically ordered distinct retrievable visual moments" in description


def test_resolution_event_schema_limits_text_to_one_temporal_moment() -> None:
    schema = KISInitialResolution.model_json_schema()
    event_schema = schema["$defs"]["KISInitialResolutionEvent"]

    text_description = (
        event_schema["properties"].get("source_text", {}).get("description")
        or event_schema["properties"].get("text", {}).get("description")
    )
    assert "retrievable chronological moment" in text_description or "distinct" in text_description


class FakeStructuredLLM:
    def __init__(self, resolution: KISInitialResolution) -> None:
        self._resolution = resolution

    def generate_structured(
        self,
        messages: Any,
        response_model: type[KISInitialResolution],
        **kwargs: Any,
    ) -> KISInitialResolution:
        assert response_model is KISInitialResolution
        return self._resolution


def test_resolver_canonicalizes_three_model_events_into_adjacent_temporal_chain() -> None:
    resolution = KISInitialResolution(
        events=[
            KISInitialResolutionEvent(source_text="A man enters a room carrying a box."),
            KISInitialResolutionEvent(source_text="The man places the box on a table."),
            KISInitialResolutionEvent(source_text="A woman opens the box on the table."),
        ]
    )

    query = (
        "A man enters a room carrying a box. "
        "The man places the box on a table. "
        "A woman opens the box on the table."
    )
    intent = KISIntentResolver(FakeStructuredLLM(resolution)).resolve_initial(
        query,
        revision=1,
    )

    assert [event.id for event in intent.events] == ["E1", "E2", "E3"]
    assert [(edge.source, edge.target) for edge in intent.temporal_edges] == [
        ("E1", "E2"),
        ("E2", "E3"),
    ]
    assert intent.entities == []


def test_resolver_keeps_one_simultaneous_scene_as_one_event() -> None:
    resolution = KISInitialResolution(
        events=[
            KISInitialResolutionEvent(text="A woman talks while holding a cup."),
        ]
    )

    intent = KISIntentResolver(FakeStructuredLLM(resolution)).resolve_initial(
        "one simultaneous scene",
        revision=1,
    )

    assert [event.id for event in intent.events] == ["E1"]
    assert intent.temporal_edges == []
