"""Tests for deterministic conversion from query text to ordered events."""

from hcmai.temporal.planner import plan_query_events, split_query_events


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


def test_planner_drops_the_trailing_question() -> None:
    """A QA question is the reviewer's job, never a moment the DP must align."""

    planned = plan_query_events(
        "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác "
        "cùng loại bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?"
    )

    assert len(planned) == 1
    assert planned[0].startswith("Hình ảnh một con cá")
    assert "bao nhiêu" not in planned[0]


def test_planner_folds_attributes_of_the_same_moment() -> None:
    """An attribute sentence joins its moment instead of demanding a second frame."""

    planned = plan_query_events(
        "Hành động cắt chùm nho bằng kéo từ giàn nho. "
        "Có thể thấy có một sợi dây màu xanh dương được buộc vào cuống."
    )

    assert len(planned) == 1
    assert planned[0].startswith("Hành động cắt chùm nho")
    assert planned[0].endswith("Có thể thấy có một sợi dây màu xanh dương được buộc vào cuống")


def test_planner_keeps_a_later_moment_out_of_attributes() -> None:
    """A forward connective marks a new moment even behind an attribute cue."""

    planned = plan_query_events(
        "Người đàn ông cầm một khối đá quý. "
        "Bên phải là một phụ nữ mỉm cười. Tiếp theo có hình ảnh toàn cảnh một mỏ đá."
    )

    assert len(planned) == 2


def test_planner_restores_timeline_order() -> None:
    """``Trước đó`` happened earlier, so monotonic DP must see it earlier."""

    planned = plan_query_events(
        "Có 4 tài xế xe ôm trong trạm xăng. Trước đó là cảnh một người đậy nắp bình xăng."
    )

    assert [event.split()[0] for event in planned] == ["Trước", "Có"]
