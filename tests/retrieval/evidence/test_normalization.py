"""Tests for per-event score normalization."""

import numpy as np
from hcmai.retrieval.evidence.normalization import minmax_rows


def test_minmax_rows_normalizes_each_event_independently() -> None:
    scores = np.array([[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]], dtype=np.float32)

    actual = minmax_rows(scores)

    np.testing.assert_allclose(actual, [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    assert actual.dtype == np.float32


def test_constant_row_becomes_zero() -> None:
    np.testing.assert_array_equal(
        minmax_rows(np.array([[7.0, 7.0]], dtype=np.float32)),
        [[0.0, 0.0]],
    )