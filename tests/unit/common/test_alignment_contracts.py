"""Tests for task-agnostic ordered event-to-frame alignment contracts."""

import pytest

from hcmai.common.schemas.alignment import AlignmentEvent, AlignmentPlan


def test_alignment_plan_requires_consecutive_event_order() -> None:
    """Prevent a caller from silently changing event positions in a plan."""

    with pytest.raises(ValueError, match="consecutive"):
        AlignmentPlan(
            events=(
                AlignmentEvent(event_id="e0", text="first", order=0),
                AlignmentEvent(event_id="e1", text="second", order=2),
            )
        )


def test_alignment_plan_is_task_agnostic() -> None:
    """Keep shared planning independent of KIS and TRAKE workflow labels."""

    plan = AlignmentPlan(
        events=(
            AlignmentEvent(event_id="e0", text="first", order=0),
            AlignmentEvent(event_id="e1", text="second", order=1),
        )
    )

    assert [event.text for event in plan.events] == ["first", "second"]
