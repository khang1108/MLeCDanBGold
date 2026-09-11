"""Unit tests for score-space residual primitives and robust normalization."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.residual import (
    robust_z,
    score_multiscale_residual,
    score_residual,
)


def test_score_residual_algebra_hand_computable() -> None:
    """Verify exact 4-term score residual on a hand-computable 2x4 matrix."""
    # scores shape: (num_events=2, num_frames=4)
    scores = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    # R(0, 1, 3) = scores[1, 3] - scores[1, 1] - scores[0, 3] + scores[0, 1]
    #            = 8.0 - 4.0 - 4.0 + 2.0 = 2.0
    result = score_residual(scores, event_index=0, a=1, b=3)
    assert isinstance(result, float)
    assert result == pytest.approx(2.0)


def test_score_residual_direction_negates_when_swapped() -> None:
    """Assert swapping endpoints a and b strictly negates the raw residual."""
    scores = np.array(
        [
            [1.5, 3.2, 2.1, 4.8],
            [0.5, 4.1, 1.9, 5.0],
        ],
        dtype=np.float64,
    )
    fwd = score_residual(scores, event_index=0, a=0, b=3)
    rev = score_residual(scores, event_index=0, a=3, b=0)
    assert fwd == pytest.approx(-rev)


def test_score_multiscale_residual_clips_to_bounds_and_takes_median() -> None:
    """Assert multiscale windows clip to array bounds and return scale median."""
    scores = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    # For a=1, b=3:
    # r=0:
    #   mean_a = [2.0, 4.0], mean_b = [4.0, 8.0]
    #   res_0 = 8.0 - 4.0 - 4.0 + 2.0 = 2.0
    # r=1:
    #   a in [0, 2]: event 0 mean=(1+2+3)/3=2.0; event 1 mean=(2+4+6)/3=4.0
    #   b in [2, 3]: event 0 mean=(3+4)/2=3.5;   event 1 mean=(6+8)/2=7.0
    #   res_1 = 7.0 - 4.0 - 3.5 + 2.0 = 1.5
    # r=2:
    #   a in [0, 3]: event 0 mean=(1+2+3+4)/4=2.5; event 1 mean=(2+4+6+8)/4=5.0
    #   b in [1, 3]: event 0 mean=(2+3+4)/3=3.0;   event 1 mean=(4+6+8)/3=6.0
    #   res_2 = 6.0 - 5.0 - 3.0 + 2.5 = 0.5
    # median([2.0, 1.5, 0.5]) = 1.5
    res = score_multiscale_residual(scores, event_index=0, a=1, b=3, radii=(0, 1, 2))
    assert isinstance(res, float)
    assert res == pytest.approx(1.5)


def test_score_multiscale_residual_radius_zero_equals_raw() -> None:
    """Radii with only radius 0 must equal raw score residual."""
    scores = np.array(
        [
            [0.2, 0.8, 0.5],
            [0.7, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    raw = score_residual(scores, 0, 0, 2)
    multi = score_multiscale_residual(scores, 0, 0, 2, radii=(0,))
    assert multi == pytest.approx(raw)


def test_robust_z_normalizes_to_zero_for_constant_input() -> None:
    """Assert constant input with zero MAD normalizes to array of zeros."""
    constant = np.array([3.14, 3.14, 3.14, 3.14], dtype=np.float64)
    normalized = robust_z(constant)
    assert np.allclose(normalized, 0.0)


def test_robust_z_expected_standardization() -> None:
    """Verify robust z-score on known symmetric distribution."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    # median = 3.0, mad = median([2, 1, 0, 1, 2]) = 1.0
    # scale = 1.4826 * 1.0
    norm = robust_z(vals)
    assert norm[2] == pytest.approx(0.0)
    assert norm[0] == pytest.approx(-norm[4])
    assert norm[1] == pytest.approx(-norm[3])
    # Pin exact magnitude to catch scale-factor regressions:
    # (1.0 - 3.0) / (1.4826 * 1.0) ≈ -1.3490
    assert norm[0] == pytest.approx(-2.0 / 1.482602218505602)


def test_robust_z_empty_array() -> None:
    """Empty input returns an empty array without error."""
    result = robust_z(np.array([], dtype=np.float64))
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_validation_non_2d_and_non_finite() -> None:
    """Reject 1D, 3D, and non-finite score matrices."""
    with pytest.raises(ValueError, match="two-dimensional"):
        score_residual(np.array([1.0, 2.0]), 0, 0, 1)

    with pytest.raises(ValueError, match="two-dimensional"):
        score_residual(np.ones((2, 2, 2)), 0, 0, 1)

    with pytest.raises(ValueError, match="finite"):
        score_residual(np.array([[1.0, np.nan], [2.0, 3.0]]), 0, 0, 1)

    with pytest.raises(ValueError, match="finite"):
        score_residual(np.array([[1.0, np.inf], [2.0, 3.0]]), 0, 0, 1)


def test_validation_invalid_event_index() -> None:
    """Reject event indices that cannot form an adjacent pair."""
    scores = np.ones((3, 4))
    with pytest.raises(ValueError, match="event_index"):
        score_residual(scores, event_index=-1, a=0, b=1)

    with pytest.raises(ValueError, match="event_index"):
        score_residual(scores, event_index=2, a=0, b=1)


def test_validation_endpoints() -> None:
    """Reject equal or out-of-range endpoints."""
    scores = np.ones((2, 4))
    with pytest.raises(ValueError, match="distinct|equal"):
        score_residual(scores, event_index=0, a=2, b=2)

    with pytest.raises(IndexError, match="out of range"):
        score_residual(scores, event_index=0, a=-1, b=2)

    with pytest.raises(IndexError, match="out of range"):
        score_residual(scores, event_index=0, a=0, b=4)


def test_validation_radii() -> None:
    """Reject empty, negative, or invalid radii."""
    scores = np.ones((2, 4))
    with pytest.raises(ValueError, match="radii"):
        score_multiscale_residual(scores, 0, 0, 1, radii=())

    with pytest.raises(ValueError, match="radii"):
        score_multiscale_residual(scores, 0, 0, 1, radii=(0, -1))


def test_validation_robust_z_eps() -> None:
    """Reject non-finite or non-positive epsilon in robust_z."""
    with pytest.raises(ValueError, match="eps"):
        robust_z(np.array([1.0, 2.0]), eps=-1e-8)

    with pytest.raises(ValueError, match="eps"):
        robust_z(np.array([1.0, 2.0]), eps=np.nan)

    with pytest.raises(ValueError, match="finite"):
        robust_z(np.array([1.0, np.nan]))
