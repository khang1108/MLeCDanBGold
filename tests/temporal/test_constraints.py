"""Tests for event-scoped interval constraints and frame masks."""

import numpy as np
import pytest

from hcmai.temporal.constraints import (
    Conditions,
    build_mask,
    merge_intervals,
    remaining_intervals,
    validate_interval,
)


def test_REQ_001_rejection_is_event_scoped():
    conditions = Conditions(
        window=(0, 40),
        confirmed=((10, 20), None),
        rejected=((), ((10, 20),)),
    )

    mask, status = build_mask(np.array([0, 10, 20, 30, 40]), conditions)

    assert status == "ready"
    np.testing.assert_array_equal(
        mask,
        [[False, True, True, False, False], [True, False, False, True, True]],
    )


def test_REQ_002_unindexed_confirmation_is_no_indexed_frames():
    conditions = Conditions(window=(0, 30), confirmed=((15, 15),), rejected=((),))

    mask, status = build_mask(np.array([10, 20]), conditions)

    assert status == "no_indexed_frames"
    assert not mask.any()


def test_REQ_003_fully_removed_confirmation_is_contradiction():
    conditions = Conditions(
        window=(0, 30),
        confirmed=((10, 20),),
        rejected=(((10, 15), (16, 20)),),
    )

    mask, status = build_mask(np.array([0, 30]), conditions)

    assert status == "contradictory_conditions"
    assert not mask.any()


@pytest.mark.parametrize("interval", [(-1, 0), (2, 1), (True, 2), (1.0, 2)])
def test_REQ_004_invalid_intervals_raise_value_error(interval):
    with pytest.raises(ValueError):
        validate_interval(interval)


def test_REQ_005_confirmation_outside_window_is_contradiction():
    conditions = Conditions(window=(10, 20), confirmed=((0, 5),), rejected=((),))

    mask, status = build_mask(np.array([10, 20]), conditions)

    assert status == "contradictory_conditions"
    assert not mask.any()


def test_REQ_005_invalid_rejection_precedes_confirmation_contradiction():
    """Validate every rejection before returning a known contradiction."""

    conditions = Conditions(
        window=(0, 10),
        confirmed=((20, 30),),
        rejected=(((True, 2),),),
    )

    with pytest.raises(ValueError):
        build_mask(np.array([0, 10]), conditions)


def test_REQ_006_closed_boundaries_are_accepted():
    conditions = Conditions(window=(10, 20), confirmed=((10, 20),), rejected=((),))

    mask, status = build_mask(np.array([10, 20]), conditions)

    assert status == "ready"
    np.testing.assert_array_equal(mask, [[True, True]])


def test_REQ_007_duplicate_rejections_are_idempotent():
    assert remaining_intervals((0, 20), ((5, 10), (5, 10))) == ((0, 4), (11, 20))


def test_REQ_008_empty_frames_have_no_indexed_frames_for_nonempty_domain():
    conditions = Conditions(window=(0, 20), confirmed=(None,), rejected=((),))

    mask, status = build_mask(np.array([], dtype=np.int64), conditions)

    assert status == "no_indexed_frames"
    assert mask.shape == (1, 0)


def test_REQ_009_merge_and_subtract_intervals():
    assert merge_intervals(((5, 7), (8, 10), (20, 20), (5, 7))) == (
        (5, 10),
        (20, 20),
    )
    assert remaining_intervals((0, 20), ((3, 5), (10, 12))) == (
        (0, 2),
        (6, 9),
        (13, 20),
    )


@pytest.mark.parametrize(
    "conditions",
    [
        Conditions(window=(0, 1), confirmed=(), rejected=()),
        Conditions(window=(0, 1), confirmed=(None,), rejected=()),
        Conditions(window=(0, 1), confirmed=(None,), rejected=((), ())),
    ],
)
def test_REQ_009_invalid_conditions_raise_value_error(conditions):
    with pytest.raises(ValueError):
        build_mask(np.array([0, 1]), conditions)


@pytest.mark.parametrize(
    "timestamps",
    [np.array([[0, 1]]), np.array([0, -1]), np.array([1, 0]), np.array([0.0, 1.0])],
)
def test_REQ_009_invalid_timestamp_shapes_raise_value_error(timestamps):
    conditions = Conditions(window=(0, 1), confirmed=(None,), rejected=((),))

    with pytest.raises(ValueError):
        build_mask(timestamps, conditions)


def test_REQ_009_mixed_contradiction_preserves_valid_row():
    conditions = Conditions(
        window=(0, 20),
        confirmed=(None, (5, 10)),
        rejected=((), ((5, 10),)),
    )

    mask, status = build_mask(np.array([0, 5, 10, 20]), conditions)

    assert status == "contradictory_conditions"
    np.testing.assert_array_equal(mask, [[True, True, True, True], [False] * 4])


def test_REQ_009_mixed_unindexed_event_reports_no_indexed_frames():
    conditions = Conditions(
        window=(0, 20),
        confirmed=(None, (7, 7)),
        rejected=((), ()),
    )

    mask, status = build_mask(np.array([0, 10, 20]), conditions)

    assert status == "no_indexed_frames"
    np.testing.assert_array_equal(mask, [[True, True, True], [False] * 3])


@pytest.mark.parametrize("interval", [(1,), (1, 2, 3), "12", None])
def test_REQ_009_malformed_interval_lengths_raise_value_error(interval):
    with pytest.raises(ValueError):
        validate_interval(interval)


def test_REQ_009_duplicate_timestamps_are_accepted_and_output_is_bool():
    conditions = Conditions(window=(0, 1), confirmed=(None,), rejected=((),))

    mask, status = build_mask(np.array([0, 0, 1], dtype=np.int64), conditions)

    assert status == "ready"
    assert mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(mask, [[True, True, True]])
