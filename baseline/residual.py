"""Pure score-space residual calculation and robust normalization primitives.

This module owns pure mathematical computations on event-to-frame similarity matrices.
It does not perform model inference, disk I/O, FAISS indexing, or DP lattice search.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

__all__ = ["score_residual", "score_multiscale_residual", "robust_z"]

# Scale factor for normal distribution consistency (1 / norm.ppf(0.75) ≈ 1.482602218505602)
_MAD_SCALE_FACTOR: float = 1.482602218505602


def _validate_score_matrix(scores: np.ndarray) -> np.ndarray:
    """Ensure scores is a finite, non-empty 2D array."""
    arr = np.asarray(scores)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("scores must be a non-empty two-dimensional matrix")
    if not np.isfinite(arr).all():
        raise ValueError("scores matrix must contain only finite numbers")
    return arr


def _validate_event_index(scores: np.ndarray, event_index: int) -> None:
    """Ensure event_index and event_index + 1 are valid rows."""
    if event_index < 0 or event_index >= scores.shape[0] - 1:
        raise ValueError(
            f"event_index {event_index} is invalid for matrix with {scores.shape[0]} event rows; "
            "must satisfy 0 <= event_index < num_events - 1"
        )


def _validate_endpoints(scores: np.ndarray, a: int, b: int) -> None:
    """Ensure a and b are distinct and within matrix column bounds."""
    num_frames = scores.shape[1]
    if a == b:
        raise ValueError(f"endpoints a and b must be distinct, got a={a}, b={b}")
    if a < 0 or a >= num_frames:
        raise IndexError(f"endpoint a={a} is out of range for scores columns (0 to {num_frames - 1})")
    if b < 0 or b >= num_frames:
        raise IndexError(f"endpoint b={b} is out of range for scores columns (0 to {num_frames - 1})")


def score_residual(
    scores: np.ndarray,
    event_index: int,
    a: int,
    b: int,
) -> float:
    """Compute the 4-term raw score-space residual between adjacent events and frame endpoints.

    Definition:
        R(i, a, b) = scores[i+1, b] - scores[i+1, a] - scores[i, b] + scores[i, a]

    Parameters
    ----------
    scores : np.ndarray
        2D matrix of shape (num_events, num_frames).
    event_index : int
        Index i of the first event (i and i+1 form the adjacent event pair).
    a : int
        Column index of the first frame candidate.
    b : int
        Column index of the second frame candidate.

    Returns
    -------
    float
        Raw directional score-space residual.
    """
    matrix = _validate_score_matrix(scores)
    _validate_event_index(matrix, event_index)
    _validate_endpoints(matrix, a, b)
    return _raw_residual(matrix, event_index, a, b)


def _raw_residual(matrix: np.ndarray, event_index: int, a: int, b: int) -> float:
    """Compute the 4-term residual on an already-validated matrix.

    Callers must ensure matrix is finite 2D, event_index is valid, and a != b
    within bounds. This avoids redundant validation in tight loops.
    """
    return float(
        matrix[event_index + 1, b]
        - matrix[event_index + 1, a]
        - matrix[event_index, b]
        + matrix[event_index, a]
    )


def score_multiscale_residual(
    scores: np.ndarray,
    event_index: int,
    a: int,
    b: int,
    radii: Sequence[int] = (0, 1, 2),
) -> float:
    """Compute the multiscale score residual across neighborhood smoothing radii.

    For each radius r in radii, replaces each endpoint score with the mean over
    keyframe neighbors in [max(0, x - r), min(N, x + r + 1)], computes the
    residual at scale r, and returns the median across all scales.

    Parameters
    ----------
    scores : np.ndarray
        2D matrix of shape (num_events, num_frames).
    event_index : int
        Index i of the first event.
    a : int
        Column index of the first frame candidate.
    b : int
        Column index of the second frame candidate.
    radii : Sequence[int], default=(0, 1, 2)
        Keyframe neighborhood radii to evaluate.

    Returns
    -------
    float
        Median residual across specified scales.
    """
    matrix = _validate_score_matrix(scores)
    _validate_event_index(matrix, event_index)
    _validate_endpoints(matrix, a, b)

    if not radii:
        raise ValueError("radii must not be empty")
    if any(r < 0 for r in radii):
        raise ValueError(f"all radii must be non-negative integers, got {radii}")

    num_frames = matrix.shape[1]
    scale_residuals: list[float] = []

    for r in radii:
        if r == 0:
            scale_residuals.append(_raw_residual(matrix, event_index, a, b))
        else:
            start_a = max(0, a - r)
            end_a = min(num_frames, a + r + 1)
            start_b = max(0, b - r)
            end_b = min(num_frames, b + r + 1)

            # Neighborhood means at scale r
            mean_0_a = float(np.mean(matrix[event_index, start_a:end_a]))
            mean_0_b = float(np.mean(matrix[event_index, start_b:end_b]))
            mean_1_a = float(np.mean(matrix[event_index + 1, start_a:end_a]))
            mean_1_b = float(np.mean(matrix[event_index + 1, start_b:end_b]))

            scale_residuals.append(mean_1_b - mean_1_a - mean_0_b + mean_0_a)

    return float(np.median(scale_residuals))


def robust_z(
    values: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Standardize an array of values using Median and Median Absolute Deviation (MAD).

    If MAD is smaller than eps (e.g. constant values), returns an array of zeros.

    Parameters
    ----------
    values : np.ndarray
        Array of numerical values to normalize.
    eps : float, default=1e-8
        Tolerance threshold below which variation is considered zero.

    Returns
    -------
    np.ndarray
        Robust z-score normalized array of float64.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be a finite positive number, got {eps}")

    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("values must contain only finite numbers")
    if arr.size == 0:
        return arr.copy()

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = _MAD_SCALE_FACTOR * mad

    if scale <= eps:
        return np.zeros_like(arr, dtype=np.float64)

    return (arr - med) / scale
