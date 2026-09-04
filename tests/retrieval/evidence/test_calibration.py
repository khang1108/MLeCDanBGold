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


@pytest.mark.parametrize("positive_count", [1, 5, 10, 20])
def test_sparse_bm25_positive_peaks_survive_with_support_mask(positive_count: int) -> None:
    """Rare BM25 exact matches must survive calibration when using a support mask."""

    raw = np.zeros((1, 1000), dtype=np.float32)
    # Set positive_count random or specific positions with positive BM25 scores (e.g. 5.0 to 10.0)
    indices = np.linspace(0, 999, positive_count, dtype=np.int64)
    raw[0, indices] = np.linspace(5.0, 10.0, positive_count, dtype=np.float32)
    support = raw > 0.0

    result = calibrate_component(
        raw,
        RobustCalibrationConfig(q_low=0.05, q_high=0.95),
        support=support,
        reliability_mode="binary",
    )

    # Positive peaks must not collapse to all zeros!
    assert result.scores.max() > 0.0
    assert result.scores[0, indices[-1]] == pytest.approx(1.0, abs=1e-5)
    # Unsupported positions remain 0
    unsupported = ~support
    assert np.all(result.scores[unsupported] == 0.0)
    # Reliability is 1.0 because at least one positive supported match exists
    assert result.reliability[0] == 1.0


def test_sparse_bm25_no_match_row_is_all_zeros() -> None:
    """A row with no BM25 matches has all zero scores and reliability zero."""

    raw = np.zeros((1, 1000), dtype=np.float32)
    support = raw > 0.0
    result = calibrate_component(
        raw,
        RobustCalibrationConfig(),
        support=support,
        reliability_mode="binary",
    )
    np.testing.assert_array_equal(result.scores, np.zeros((1, 1000), dtype=np.float32))
    assert result.reliability[0] == 0.0


def test_equal_strength_sparse_bm25_matches_survive() -> None:
    """Equal positive lexical matches remain calibrated as exact evidence."""

    raw = np.asarray([[0.0, 0.0, 5.0, 0.0, 5.0, 0.0]], dtype=np.float32)
    support = raw > 0.0

    result = calibrate_component(
        raw,
        RobustCalibrationConfig(),
        support=support,
        reliability_mode="binary",
    )

    np.testing.assert_array_equal(
        result.scores,
        np.asarray([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    )
    assert result.reliability[0] == 1.0


def test_sparse_asr_quantiles_use_covered_frames_only() -> None:
    """ASR calibration quantiles and reliability ignore uncovered zero background."""

    raw = np.zeros((1, 1000), dtype=np.float32)
    coverage = np.zeros((1, 1000), dtype=bool)
    # 20 covered frames with scores in [0.2, 0.8]
    cov_indices = np.arange(100, 120, dtype=np.int64)
    coverage[0, cov_indices] = True
    raw[0, cov_indices] = np.linspace(0.2, 0.8, 20, dtype=np.float32)

    result = calibrate_component(
        raw,
        RobustCalibrationConfig(),
        support=coverage,
        reliability_mode="contrast",
    )

    # Uncovered positions must be 0.0
    assert np.all(result.scores[0, ~coverage[0]] == 0.0)
    # Top covered score should be calibrated to 1.0
    assert result.scores[0, cov_indices[-1]] == pytest.approx(1.0, abs=1e-3)
    # Reliability must be non-zero since covered frames show clear contrast
    assert result.reliability[0] > 0.0


def test_top_k_is_bounded_by_top_k_max() -> None:
    """top_k_max bounds the number of top frames considered for reliability."""

    raw = np.random.RandomState(42).randn(1, 20000).astype(np.float32)
    cfg = RobustCalibrationConfig(top_fraction=0.1, top_k_max=64)
    result = calibrate_component(raw, cfg)
    assert 0.0 <= result.reliability[0] <= 1.0
