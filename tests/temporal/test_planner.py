"""Tests for deterministic conversion from query text to ordered events."""

from hcmai.temporal.planner import split_query_events


def test_multiline_query_prefers_lines() -> None:
    """Prefer explicit newline-separated events over sentence splitting."""

    assert split_query_events("hold ingredient\nroll ingredient\ncoat flour") == (
        "hold ingredient",
        "roll ingredient",
        "coat flour",
    )


def test_single_line_query_splits_sentences() -> None:
    """Use sentence boundaries when one line clearly contains multiple events."""

    assert split_query_events("Hold ingredient. Roll ingredient! Coat flour?") == (
        "Hold ingredient",
        "Roll ingredient",
        "Coat flour",
    )


def test_single_event_stays_single() -> None:
    """Avoid inventing multiple events for an ordinary one-event query."""

    assert split_query_events("chef holds a bowl") == ("chef holds a bowl",)
