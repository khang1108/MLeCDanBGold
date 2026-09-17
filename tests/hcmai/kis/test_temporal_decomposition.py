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
