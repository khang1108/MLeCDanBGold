"""Tests for deterministic conversion from query text to ordered events."""

from hcmai.temporal.planner import split_query_events


def test_multiline_query_becomes_ordered_events() -> None:
    """Treat newline-separated query steps as an explicit simple plan."""

    assert split_query_events("first event\nsecond event\nthird event") == (
        "first event",
        "second event",
        "third event",
    )


def test_sentence_query_becomes_ordered_events() -> None:
    """Use sentence boundaries only when they yield multiple nonempty events."""

    assert split_query_events(
        "First action. Then second action. Finally third action."
    ) == ("First action", "Then second action", "Finally third action")


def test_single_sentence_remains_one_event() -> None:
    """Avoid inventing an event sequence for an ordinary single-event KIS query."""

    assert split_query_events("person splashes water on face") == (
        "person splashes water on face",
    )
