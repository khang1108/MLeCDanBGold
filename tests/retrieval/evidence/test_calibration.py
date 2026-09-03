"""Tests for robust row-wise score calibration and reliability."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.common.config import RobustCalibrationConfig
from hcmai.retrieval.evidence.calibration import (
    CalibratedComponent,
    calibrate_component,
)


def test_constant_row_has_zero_scores_and_zero_reliability() -> None:
    """A row with identical values produces zero scores and zero reliability."""

    result = calibrate_component(
        np.asarray([[0.3, 0.3, 0.3, 0.3]], dtype=np.float32),
        RobustCalibrationConfig(),
    )
    np.testing.assert_array_equal(result.scores, np.zeros((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(result.reliability, np.asarray([0.0], dtype=np.float32))


def test_tiny_dynamic_range_is_ranked_but_not_fully_trusted() -> None:
    """A row with very small differences preserves order with low reliability."""

    result = calibrate_component(
        np.asarray([[0.201, 0.203, 0.204, 0.209]], dtype=np.float32),
        RobustCalibrationConfig(),
    )
    assert result.scores[0, -1] > result.scores[0, 0]
    assert 0.0 < result.reliability[0] < 1.0


def test_large_outlier_is_clipped_by_quantiles() -> None:
    """Outliers above q_high are clipped at 1.0 without distorting the bulk."""

    raw = np.asarray([[0.0, 0.1, 0.2, 0.3, 100.0]], dtype=np.float32)
    result = calibrate_component(raw, RobustCalibrationConfig(q_high=0.8))
    assert result.scores.max() == 1.0
    assert np.isfinite(result.scores).all()


def test_positive_affine_rescaling_preserves_calibration() -> None:
    """Scaling and shifting raw scores preserves calibration scores and reliability."""

    raw = np.asarray([[0.2, 0.4, 0.9, 1.2]], dtype=np.float32)
    a = calibrate_component(raw, RobustCalibrationConfig())
    b = calibrate_component(raw * 100.0 + 7.0, RobustCalibrationConfig())

    np.testing.assert_allclose(a.scores, b.scores, atol=1e-6)
    np.testing.assert_allclose(a.reliability, b.reliability, atol=1e-6)


def test_empty_frames_returns_zero_scores_and_zero_reliability() -> None:
    """Zero columns in raw_scores returns empty calibrated component."""

    raw = np.zeros((2, 0), dtype=np.float32)
    result = calibrate_component(raw, RobustCalibrationConfig())
    assert result.scores.shape == (2, 0)
    assert result.reliability.shape == (2,)
    np.testing.assert_array_equal(result.reliability, [0.0, 0.0])


def test_invalid_raw_scores_raise_value_error() -> None:
    """Non-2D or non-finite inputs raise ValueError."""

    with pytest.raises(ValueError, match="two-dimensional"):
        calibrate_component(np.asarray([1.0, 2.0]), RobustCalibrationConfig())

    with pytest.raises(ValueError, match="finite"):
        calibrate_component(np.asarray([[1.0, np.nan]]), RobustCalibrationConfig())
