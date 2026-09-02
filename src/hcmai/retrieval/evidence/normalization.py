"""Numerical normalization for full-corpus temporal evidence."""

from __future__ import annotations

import numpy as np


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    """Normalize each event row independently; constant rows become zero."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    if values.shape[1] == 0:
        return np.zeros_like(values)
    minimum = values.min(axis=1, keepdims=True)
    span = values.max(axis=1, keepdims=True) - minimum
    normalized = np.zeros_like(values)
    np.divide(values - minimum, span, out=normalized, where=span > 0)
    return normalized