"""Robust quantile-based calibration and reliability estimation for evidence rows.

This module normalizes raw score components across frames and derives a
continuous reliability score in [0, 1] per event. Weak or near-constant
evidence rows retain their internal ranking but receive lower reliability,
preventing noise from dominating fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
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
    *,
    support: np.ndarray | None = None,
    reliability_mode: Literal["contrast", "binary"] = "contrast",
) -> CalibratedComponent:
    """Calibrate a 2D raw score matrix and calculate row-wise reliability.

    Uses quantile clipping [q_low, q_high] on supported frames to mitigate
    extreme outliers, followed by span normalization. Reliability is derived
    either from contrast between top-k and median (for continuous modalities)
    or binary match presence (for sparse lexical matches).

    Args:
        raw_scores: 2D array of shape ``[event_count, frame_count]`` of finite
            floating-point scores.
        config: Calibration hyperparameters controlling quantiles, top fraction,
            top_k_max, and epsilon.
        support: Optional boolean mask of shape ``[event_count, frame_count]``
            defining which frame positions should inform calibration statistics.
        reliability_mode: Mode for calculating reliability: ``"contrast"`` or ``"binary"``.

    Returns:
        A :class:`CalibratedComponent` containing calibrated scores and
        per-event reliability.

    Raises:
        ValueError: If ``raw_scores`` is not 2-dimensional, contains non-finite
            values, or ``support`` shape does not match.
    """

    values = np.asarray(raw_scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("raw_scores must be two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("raw_scores must contain only finite values")

    event_count, frame_count = values.shape
    if frame_count == 0:
        return CalibratedComponent(
            np.zeros_like(values),
            np.zeros(event_count, dtype=np.float32),
        )

    if support is not None:
        support_mask = np.asarray(support, dtype=bool)
        if support_mask.shape != values.shape:
            raise ValueError(
                f"support mask shape {support_mask.shape} does not match raw_scores {values.shape}"
            )
    else:
        support_mask = None

    calibrated = np.zeros_like(values, dtype=np.float32)
    reliability = np.zeros(event_count, dtype=np.float32)

    for e in range(event_count):
        row = values[e]
        if np.ptp(row) <= config.eps:
            calibrated[e].fill(0.0)
            reliability[e] = 0.0
            continue

        if support_mask is not None:
            row_support = support_mask[e]
            supported_indices = np.flatnonzero(row_support)
            supported_vals = row[supported_indices]
        else:
            supported_indices = np.arange(frame_count)
            supported_vals = row

        valid_count = len(supported_vals)
        if valid_count == 0:
            calibrated[e].fill(0.0)
            reliability[e] = 0.0
            continue

        if valid_count == 1:
            idx = supported_indices[0]
            val = float(supported_vals[0])
            calibrated[e, idx] = 1.0 if val > 0.0 else 0.0
            reliability[e] = 1.0 if (reliability_mode == "binary" or val > 0.0) else 0.0
            continue

        low = float(np.quantile(supported_vals, config.q_low))
        high = float(np.quantile(supported_vals, config.q_high))
        span = high - low

        if span > config.eps:
            clipped = np.clip(supported_vals, low, high)
            calibrated[e, supported_indices] = (clipped - low) / span
        elif reliability_mode == "binary" and np.any(supported_vals > 0.0):
            # Equal-strength lexical hits are still exact evidence. Their lack
            # of internal contrast must not erase every supported match.
            calibrated[e, supported_indices] = 1.0
        else:
            calibrated[e, supported_indices] = 0.0

        if reliability_mode == "binary":
            reliability[e] = 1.0 if np.any(supported_vals > 0.0) else 0.0
        else:
            if span <= config.eps:
                reliability[e] = 0.0
            else:
                top_k = min(
                    config.top_k_max,
                    max(1, int(np.ceil(valid_count * config.top_fraction))),
                )
                median = float(np.median(supported_vals))
                q25 = float(np.quantile(supported_vals, 0.25))
                q75 = float(np.quantile(supported_vals, 0.75))
                iqr = q75 - q25
                top_mean = float(np.mean(np.partition(supported_vals, -top_k)[-top_k:]))
                robust_z = max(top_mean - median, 0.0) / (iqr + config.eps)
                rel = robust_z / (1.0 + robust_z)
                reliability[e] = float(np.clip(rel, 0.0, 1.0))

    return CalibratedComponent(
        np.asarray(calibrated, dtype=np.float32),
        np.asarray(reliability, dtype=np.float32),
    )


__all__ = ["CalibratedComponent", "calibrate_component"]
