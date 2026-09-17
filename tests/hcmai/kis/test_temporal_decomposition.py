"""Focused contract and regression tests for KIS initial temporal decomposition.

This test module verifies the prompt and structured-output schema contracts that
prevent the initial resolver from collapsing multi-moment natural language narratives
into a single event object.
"""

from hcmai.kis.prompts import (
    KIS_RESOLVER_SYSTEM_PROMPT,
    build_kis_intent_messages,
)


def test_initial_prompt_states_one_clue_can_produce_multiple_events() -> None:
    messages = build_kis_intent_messages(
        ["A person enters a room. Then the person sits down."]
    )

    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "A clue is not an event" in system
    assert "one clue may resolve to one or many chronological events" in system
    assert "single narrative containing multiple chronological moments" in user
    assert "One clue may produce multiple events" in user


def test_initial_prompt_requires_sequential_moments_to_be_separate_events() -> None:
    system = KIS_RESOLVER_SYSTEM_PROMPT

    assert "TEMPORAL DECOMPOSITION" in system
    assert "Each distinct sequential moment MUST be a separate object" in system
    assert "Do NOT combine several sequential moments into a single event text" in system
    assert "Merge actions only when they occur as part of the same temporal moment" in system


def test_initial_prompt_contains_non_cycling_one_clue_multi_event_example() -> None:
    system = KIS_RESOLVER_SYSTEM_PROMPT

    assert "A man first enters a room carrying a box" in system
    assert "Then he puts the box on a table" in system
    assert "Finally a woman opens the box" in system
    assert '"events": [' in system
    assert system.count('"text":') >= 3


def test_resolution_schema_describes_events_as_distinct_temporal_moments() -> None:
    from hcmai.kis.models import KISResolution

    schema = KISResolution.model_json_schema()
    events = schema["properties"]["events"]

    description = events["description"]
    assert "One input clue may produce multiple events" in description
    assert "sequential moments must be separate list items" in description


def test_resolution_event_schema_limits_text_to_one_temporal_moment() -> None:
    from hcmai.kis.models import KISResolution

    schema = KISResolution.model_json_schema()
    event_schema = schema["$defs"]["KISResolutionEvent"]

    text_description = event_schema["properties"]["text"]["description"]
    indices_description = event_schema["properties"]["entity_indices"]["description"]

    assert "exactly one temporal moment" in text_description
    assert "Never combine distinct sequential moments" in text_description
    assert "Zero-based indices into the entities array" in indices_description
    assert "same entity index may appear in multiple events" in indices_description

