"""Tests for revisioned single-video temporal exploration and undo."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hcmai.orchestration.workflows.temporal_exploration import (
    ExplorationConflict,
    ExplorationUnavailable,
    QueryBinding,
    TemporalExploration,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import AlignedPath


def _video() -> VideoEventScores:
    """Build one scorer-owned matrix for two ordered events."""

    return VideoEventScores(
        video_id="v",
        frame_ids=np.asarray(["a", "b", "c"], dtype=object),
        frame_idx=np.asarray([1, 2, 3]),
        timestamps_ms=np.asarray([10, 20, 30]),
        scores=np.ones((2, 3)),
    )


def _binding() -> QueryBinding:
    """Build one valid immutable event/scoring binding."""

    return QueryBinding(
        "A then B",
        "events-1",
        ("A", "B"),
        ("A", "B"),
        None,
        True,
        False,
        "scores-1",
    )


def _path(*, first_timestamp: int = 10) -> AlignedPath:
    """Build a canonical two-event path returned by a fake decoder."""

    return AlignedPath(
        video_id="v",
        score=2.0,
        frame_ids=("a", "b"),
        frame_idxs=(1, 2),
        timestamps_ms=(first_timestamp, 20),
    )


def _new_branch(
    *,
    source: VideoEventScores | None = None,
) -> tuple[TemporalExploration, SimpleNamespace, VideoEventScores]:
    """Open one branch over a deterministic fake scorer and decoder."""

    video = source or _video()
    service = SimpleNamespace(
        score_videos=Mock(return_value=((video,), 1.0)),
        decode_video=Mock(return_value=(_path(),)),
    )
    branch = TemporalExploration(service)
    branch.open(_binding(), "v", (0, 40))
    return branch, service, video


def _apply(branch: TemporalExploration, action: str, **kwargs: object):
    """Apply feedback against the branch's current binding and revision."""

    return branch.apply(
        expected_revision=branch.current().revision,
        event_version="events-1",
        scoring_revision="scores-1",
        action=action,
        **kwargs,
    )


def test_feedback_then_undo_reuses_scores_and_preserves_source() -> None:
    """Feedback and undo decode cached scores without rescoring or mutation."""

    branch, service, source = _new_branch()
    original = source.scores.copy()
    initial = branch.current()

    changed = _apply(branch, "confirm", event_index=0, interval=(10, 20))
    restored = branch.undo(
        expected_revision=changed.revision,
        event_version="events-1",
        scoring_revision="scores-1",
    )

    assert restored.conditions == initial.conditions
    assert restored.revision > changed.revision
    assert service.score_videos.call_count == 1
    assert service.decode_video.call_count == 3
    np.testing.assert_array_equal(source.scores, original)


def test_open_forwards_binding_and_freezes_separate_array_copies() -> None:
    """Retain only detached, read-only arrays for the selected video."""

    branch, service, source = _new_branch()

    service.score_videos.assert_called_once_with(
        ("A", "B"),
        retrieval_events=("A", "B"),
        caption_events=None,
        use_dense=True,
        use_bm25=False,
    )
    for name in ("frame_ids", "frame_idx", "timestamps_ms", "scores"):
        copied = getattr(branch._video, name)
        original = getattr(source, name)
        assert copied is not original
        assert not np.shares_memory(copied, original)
        assert copied.flags.writeable is False
        assert original.flags.writeable is True


def test_open_publishes_initial_conditions_and_revision() -> None:
    """Start at revision one with an explicit window and no feedback history."""

    branch, _, _ = _new_branch()

    view = branch.current()

    assert view.revision == 1
    assert view.event_version == "events-1"
    assert view.video_id == "v"
    assert view.conditions.window == (0, 40)
    assert view.conditions.confirmed == (None, None)
    assert view.conditions.rejected == ((), ())
    assert view.status == "ok"
    assert view.can_undo is False


@pytest.mark.parametrize(
    ("revision", "event_version", "scoring_revision", "message"),
    [
        (0, "events-1", "scores-1", "revision"),
        (1, "events-2", "scores-1", "event"),
        (1, "events-1", "scores-2", "scoring"),
    ],
)
def test_stale_guards_precede_feedback_validation_and_leave_state_unchanged(
    revision: int,
    event_version: str,
    scoring_revision: str,
    message: str,
) -> None:
    """Reject stale requests before validating or evaluating their payload."""

    branch, service, _ = _new_branch()
    before = branch.current()
    decode_count = service.decode_video.call_count

    with pytest.raises(ExplorationConflict, match=message):
        branch.apply(
            expected_revision=revision,
            event_version=event_version,
            scoring_revision=scoring_revision,
            action="invalid",
        )

    assert branch.current() == before
    assert service.score_videos.call_count == 1
    assert service.decode_video.call_count == decode_count


def test_stale_undo_guard_leaves_checkpoint_untouched() -> None:
    """Reject an old undo revision without consuming retained conditions."""

    branch, service, _ = _new_branch()
    changed = _apply(branch, "confirm", event_index=0, interval=(10, 10))
    history_before = tuple(branch._history)
    decode_count = service.decode_video.call_count

    with pytest.raises(ExplorationConflict, match="revision"):
        branch.undo(
            expected_revision=changed.revision - 1,
            event_version="events-1",
            scoring_revision="scores-1",
        )

    assert branch.current() == changed
    assert tuple(branch._history) == history_before
    assert service.decode_video.call_count == decode_count


def test_reconfirm_then_undo_restores_previous_point() -> None:
    """Undo reconfirmation to the immediately preceding confirmation."""

    branch, _, _ = _new_branch()

    _apply(branch, "confirm", event_index=0, interval=(10, 10))
    reconfirmed = _apply(branch, "confirm", event_index=0, interval=(20, 20))
    restored = branch.undo(
        expected_revision=reconfirmed.revision,
        event_version="events-1",
        scoring_revision="scores-1",
    )

    assert restored.conditions.confirmed == ((10, 10), None)


def test_reject_union_duplicate_noop_is_event_scoped() -> None:
    """Merge one event's rejections and ignore an equivalent duplicate."""

    branch, service, _ = _new_branch()

    _apply(branch, "reject", event_index=0, interval=(10, 15))
    merged = _apply(branch, "reject", event_index=0, interval=(16, 20))
    decode_count = service.decode_video.call_count
    duplicate = _apply(branch, "reject", event_index=0, interval=(10, 20))

    assert merged.conditions.rejected == (((10, 20),), ())
    assert duplicate == merged
    assert service.decode_video.call_count == decode_count


def test_unknown_is_not_negative_and_does_not_decode_or_revise() -> None:
    """Keep unknown feedback informational rather than treating it as reject."""

    branch, service, _ = _new_branch()
    before = branch.current()
    decode_count = service.decode_video.call_count

    after = _apply(branch, "unknown", event_index=1)

    assert after == before
    assert after.conditions.rejected == ((), ())
    assert service.decode_video.call_count == decode_count


def test_semantic_failure_commits_conditions_for_undo() -> None:
    """Retain contradictory feedback as a revision the user can undo."""

    branch, service, _ = _new_branch()
    before = branch.current()
    decode_count = service.decode_video.call_count

    failed = _apply(branch, "confirm", event_index=0, interval=(50, 50))

    assert failed.status == "contradictory_conditions"
    assert failed.conditions.confirmed == ((50, 50), None)
    assert failed.revision == before.revision + 1
    assert failed.can_undo is True
    assert service.decode_video.call_count == decode_count


@pytest.mark.parametrize(
    ("action", "event_index", "interval"),
    [
        ("invalid", None, None),
        ([], None, None),
        ("confirm", None, (0, 1)),
        ("reject", 2, (0, 1)),
        ("unknown", -1, None),
        ("unknown", True, None),
        ("window", 0, (0, 1)),
        ("confirm", 0, None),
        ("reject", 0, None),
        ("window", None, None),
        ("unknown", 0, (0, 1)),
        ("confirm", 0, (2, 1)),
    ],
)
def test_invalid_action_arguments_do_not_mutate(
    action: str,
    event_index: int | None,
    interval: tuple[int, int] | None,
) -> None:
    """Reject missing, extra, malformed, or incompatible action arguments."""

    branch, service, _ = _new_branch()
    before = branch.current()
    decode_count = service.decode_video.call_count

    with pytest.raises(ValueError):
        _apply(
            branch,
            action,
            event_index=event_index,
            interval=interval,
        )

    assert branch.current() == before
    assert service.decode_video.call_count == decode_count


@pytest.mark.parametrize(
    "binding",
    [
        replace(_binding(), query="  "),
        replace(_binding(), event_version=""),
        replace(_binding(), scoring_revision=" "),
        replace(_binding(), events=()),
        replace(_binding(), events=(" A", "B")),
        replace(_binding(), retrieval_events=("A",)),
        replace(_binding(), retrieval_events=("A", " B")),
        replace(_binding(), caption_events=("A",)),
        replace(_binding(), use_dense=False, use_bm25=False),
    ],
)
def test_invalid_binding_is_rejected_before_scoring(binding: QueryBinding) -> None:
    """Validate an immutable normalized scoring binding before model work."""

    service = SimpleNamespace(score_videos=Mock(), decode_video=Mock())
    branch = TemporalExploration(service)

    with pytest.raises(ValueError):
        branch.open(binding, "v", (0, 40))

    service.score_videos.assert_not_called()


@pytest.mark.parametrize(
    "video_id, window",
    [("", (0, 40)), ("  ", (0, 40)), ("v", (2, 1))],
)
def test_invalid_open_arguments_are_rejected_before_scoring(
    video_id: str,
    window: tuple[int, int],
) -> None:
    """Validate selected identity and closed interval before acquisition."""

    service = SimpleNamespace(score_videos=Mock(), decode_video=Mock())
    branch = TemporalExploration(service)

    with pytest.raises(ValueError):
        branch.open(_binding(), video_id, window)

    service.score_videos.assert_not_called()


def test_open_rejects_active_branch_and_missing_selected_video() -> None:
    """Require explicit close and distinguish absent score data from evidence."""

    branch, service, _ = _new_branch()
    before = branch.current()

    with pytest.raises(ValueError, match="active"):
        branch.open(_binding(), "v", (0, 40))
    assert branch.current() == before
    assert service.score_videos.call_count == 1

    empty_service = SimpleNamespace(
        score_videos=Mock(return_value=((), 1.0)),
        decode_video=Mock(),
    )
    unopened = TemporalExploration(empty_service)
    with pytest.raises(ExplorationUnavailable, match="video"):
        unopened.open(_binding(), "v", (0, 40))
    with pytest.raises(ExplorationUnavailable):
        unopened.current()


def test_close_checks_revision_then_releases_branch_state() -> None:
    """Keep an active branch on stale close, then release it on valid close."""

    branch, _, _ = _new_branch()
    before = branch.current()

    with pytest.raises(ExplorationConflict, match="revision"):
        branch.close(expected_revision=0)
    assert branch.current() == before

    branch.close(expected_revision=before.revision)
    with pytest.raises(ExplorationUnavailable):
        branch.current()
    with pytest.raises(ExplorationUnavailable):
        branch.close(expected_revision=before.revision)


def test_undo_with_empty_history_is_noop() -> None:
    """Avoid reevaluation and revision changes when nothing can be undone."""

    branch, service, _ = _new_branch()
    before = branch.current()
    decode_count = service.decode_video.call_count

    after = branch.undo(
        expected_revision=before.revision,
        event_version="events-1",
        scoring_revision="scores-1",
    )

    assert after == before
    assert service.decode_video.call_count == decode_count


def test_two_branches_from_one_source_are_isolated() -> None:
    """Keep branch conditions and score copies isolated from shared scoring."""

    source = _video()
    first, _, _ = _new_branch(source=source)
    second, _, _ = _new_branch(source=source)
    second_before = second.current()

    _apply(first, "confirm", event_index=0, interval=(10, 10))

    assert second.current() == second_before
    assert first._video.scores is not second._video.scores
    np.testing.assert_array_equal(source.scores, np.ones((2, 3)))


def test_known_infrastructure_failure_is_transactional_and_unavailable() -> None:
    """Keep revision, conditions, and history when evaluation infrastructure fails."""

    branch, service, _ = _new_branch()
    before = branch.current()
    history_before = tuple(branch._history)
    service.decode_video.side_effect = OSError("materializer unavailable")

    with pytest.raises(ExplorationUnavailable) as captured:
        _apply(branch, "confirm", event_index=0, interval=(10, 10))

    assert isinstance(captured.value.__cause__, OSError)
    assert branch.current() == before
    assert tuple(branch._history) == history_before


def test_programming_error_is_not_swallowed_or_committed() -> None:
    """Propagate unexpected decoder bugs without publishing partial state."""

    branch, service, _ = _new_branch()
    before = branch.current()
    service.decode_video.side_effect = AttributeError("bug")

    with pytest.raises(AttributeError, match="bug"):
        _apply(branch, "confirm", event_index=0, interval=(10, 10))

    assert branch.current() == before


def test_failed_open_evaluation_never_publishes_partial_state() -> None:
    """Publish active state only after selected-video evaluation succeeds."""

    source = _video()
    service = SimpleNamespace(
        score_videos=Mock(return_value=((source,), 1.0)),
        decode_video=Mock(side_effect=TimeoutError("backend timeout")),
    )
    branch = TemporalExploration(service)

    with pytest.raises(ExplorationUnavailable):
        branch.open(_binding(), "v", (0, 40))
    with pytest.raises(ExplorationUnavailable):
        branch.current()


def test_failed_open_scoring_maps_known_io_error_without_state() -> None:
    """Map acquisition I/O failure while leaving the branch unopened."""

    service = SimpleNamespace(
        score_videos=Mock(side_effect=OSError("index missing")),
        decode_video=Mock(),
    )
    branch = TemporalExploration(service)

    with pytest.raises(ExplorationUnavailable) as captured:
        branch.open(_binding(), "v", (0, 40))

    assert isinstance(captured.value.__cause__, OSError)
    service.decode_video.assert_not_called()
    with pytest.raises(ExplorationUnavailable):
        branch.current()
