"""Robust quantile-based calibration and reliability estimation for evidence rows.

This module normalizes raw score components across frames and derives a
continuous reliability score in [0, 1] per event. Weak or near-constant
evidence rows retain their internal ranking but receive lower reliability,
preventing noise from dominating fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from hcmai.common.config import RobustCalibrationConfig


@dataclass(frozen=True, slots=True)
class CalibratedComponent:
    """Calibrated evidence matrix and per-event reliability.

    Attributes:
        scores: Float32 matrix of shape ``[event_count, frame_count]`` in
            ``[0, 1]`` after quantile clipping and min-max scaling.
        reliability: Float32 array of shape ``[event_count]`` in ``[0, 1]``
            measuring signal prominence over background noise.
    """

    scores: np.ndarray
    reliability: np.ndarray


def calibrate_component(
    raw_scores: np.ndarray,
    config: RobustCalibrationConfig,
) -> CalibratedComponent:
    """Calibrate a 2D raw score matrix and calculate row-wise reliability.

    Uses quantile clipping [q_low, q_high] to mitigate extreme outliers,
    followed by span normalization. Reliability is derived from the robust
    z-score of top candidate frames relative to the median and IQR.

    Args:
        raw_scores: 2D array of shape ``[event_count, frame_count]`` of finite
            floating-point scores.
        config: Calibration hyperparameters controlling quantiles, top fraction,
            and epsilon.

    Returns:
        A :class:`CalibratedComponent` containing calibrated scores and
        per-event reliability.

    Raises:
        ValueError: If ``raw_scores`` is not 2-dimensional or contains non-finite
            values.
    """

    values = np.asarray(raw_scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("raw_scores must be two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_scores must contain only finite values")
    if values.shape[1] == 0:
        return CalibratedComponent(
            np.zeros_like(values),
            np.zeros(values.shape[0], dtype=np.float32),
        )

    low = np.quantile(values, config.q_low, axis=1, keepdims=True)
    high = np.quantile(values, config.q_high, axis=1, keepdims=True)
    span = high - low

    clipped = np.clip(values, low, high)
    calibrated = np.zeros_like(values, dtype=np.float32)
    np.divide(clipped - low, span, out=calibrated, where=span > config.eps)

    median = np.median(values, axis=1)
    q25 = np.quantile(values, 0.25, axis=1)
    q75 = np.quantile(values, 0.75, axis=1)
    iqr = q75 - q25
    top_k = max(1, int(np.ceil(values.shape[1] * config.top_fraction)))
    top_mean = np.mean(np.partition(values, -top_k, axis=1)[:, -top_k:], axis=1)
    robust_z = np.maximum(top_mean - median, 0.0) / (iqr + config.eps)
    reliability = robust_z / (1.0 + robust_z)
    reliability = np.where(span[:, 0] > config.eps, reliability, 0.0)

    return CalibratedComponent(
        np.asarray(calibrated, dtype=np.float32),
        np.asarray(np.clip(reliability, 0.0, 1.0), dtype=np.float32),
    )


__all__ = ["CalibratedComponent", "calibrate_component"]
