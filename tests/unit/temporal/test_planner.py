"""Tests for deterministic conversion from query text to ordered events."""

from hcmai.temporal.planner import build_alignment_plan


def _texts(plan) -> list[str]:
    """Return event text in planned order for concise test assertions."""

    return [event.text for event in plan.events]


def test_explicit_events_are_authoritative() -> None:
    """Never let query sentence parsing rewrite caller-supplied event order."""

    plan = build_alignment_plan(
        "ignored for event splitting",
        [" chef holds skewer ", "chef coats it"],
    )

    assert _texts(plan) == ["chef holds skewer", "chef coats it"]


def test_multiline_query_becomes_ordered_events() -> None:
    """Treat newline-separated query steps as an explicit simple plan."""

    plan = build_alignment_plan("first event\nsecond event\nthird event")

    assert _texts(plan) == ["first event", "second event", "third event"]


def test_sentence_query_becomes_ordered_events() -> None:
    """Use sentence boundaries only when they yield multiple nonempty events."""

    plan = build_alignment_plan(
        "First action. Then second action. Finally third action."
    )

    assert _texts(plan) == [
        "First action",
        "Then second action",
        "Finally third action",
    ]


def test_single_sentence_remains_one_event() -> None:
    """Avoid inventing an event sequence for an ordinary single-event KIS query."""

    plan = build_alignment_plan("person splashes water on face")

    assert _texts(plan) == ["person splashes water on face"]
