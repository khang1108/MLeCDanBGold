"""Tests for sparse unary candidate-lattice baseline methods."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.contracts import MethodOptions
from baseline.methods.framewise import DanteStyleUnaryDPMethod
from baseline.methods.lattice import (
    UnaryBoundedOrderMethod,
    UnaryCandidateLatticeMethod,
    candidate_orders,
    select_event_candidates,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores


def _video(video_id: str, scores: list[list[float]]) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    frame_count = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray([f"{video_id}-f{i}" for i in range(frame_count)]),
        frame_idx=np.arange(frame_count, dtype=np.int64),
        timestamps_ms=np.arange(frame_count, dtype=np.int64) * 1_000,
        scores=matrix,
    )


def test_select_event_candidates_is_deterministic_separated_and_pure() -> None:
    scores = np.asarray([[0.9, 0.8, 0.7, 0.6], [0.5, 0.5, 0.1, 0.0]])
    original = scores.copy()
    timestamps = np.asarray([0, 1_000, 5_000, 9_000])

    selected = select_event_candidates(
        scores,
        timestamps,
        limit=3,
        min_separation_ms=3_000,
    )

    assert [row.tolist() for row in selected] == [[0, 2, 3], [0, 2, 3]]
    np.testing.assert_array_equal(scores, original)


def test_lattice_matches_full_dp_when_every_frame_is_admitted() -> None:
    video = _video(
        "v1",
        [
            [9.0, 1.0, 0.0, 0.0],
            [0.0, 8.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 7.0],
        ],
    )
    options = MethodOptions(lambda_gap=0.001, candidates_per_event=4)

    full = DanteStyleUnaryDPMethod(options).rank([video], top_k=1)[0]
    lattice = UnaryCandidateLatticeMethod(options).rank([video], top_k=1)[0]

    assert lattice.frame_ids == full.frame_ids
    assert lattice.score == pytest.approx(full.score)


def test_lattice_rejects_video_without_complete_increasing_path() -> None:
    impossible = _video("v1", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    possible = _video("v2", [[1.0, 0.0], [0.0, 1.0]])
    method = UnaryCandidateLatticeMethod(MethodOptions(lambda_gap=0.0, candidates_per_event=1))

    paths = method.rank([impossible, possible], top_k=10)

    assert [path.video_id for path in paths] == ["v2"]


def test_candidate_orders_are_bounded_and_deterministic() -> None:
    assert candidate_orders(3, max_adjacent_swaps=1, max_orders=10) == (
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
    )
    assert candidate_orders(3, max_adjacent_swaps=2, max_orders=2) == (
        (0, 1, 2),
        (1, 0, 2),
    )


def test_identity_only_bounded_order_equals_strict_lattice() -> None:
    video = _video("v1", [[1.0, 0.2, 0.0], [0.0, 0.4, 1.0]])
    strict_options = MethodOptions(lambda_gap=0.0, candidates_per_event=3)
    bounded_options = MethodOptions(
        lambda_gap=0.0,
        candidates_per_event=3,
        max_adjacent_swaps=0,
    )

    strict = UnaryCandidateLatticeMethod(strict_options).rank([video], top_k=1)[0]
    bounded = UnaryBoundedOrderMethod(bounded_options).rank([video], top_k=1)[0]

    assert bounded == strict


def test_one_adjacent_swap_recovers_inversion_in_original_event_order() -> None:
    video = _video("v1", [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    options = MethodOptions(
        lambda_gap=0.0,
        candidates_per_event=1,
        max_adjacent_swaps=1,
    )

    path = UnaryBoundedOrderMethod(options).rank([video], top_k=1)[0]

    assert path.frame_ids == ("v1-f2", "v1-f0")
    assert path.chronological_event_order == (1, 0)
    assert path.score == pytest.approx(2.0)


def test_swap_penalty_can_preserve_identity_order() -> None:
    video = _video("v1", [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]])
    options = MethodOptions(
        lambda_gap=0.0,
        candidates_per_event=3,
        max_adjacent_swaps=1,
        swap_penalty=2.0,
    )

    path = UnaryBoundedOrderMethod(options).rank([video], top_k=1)[0]

    assert path.chronological_event_order == (0, 1)
    assert path.score == pytest.approx(0.5)


def test_lattice_validates_candidate_arguments() -> None:
    with pytest.raises(ValueError, match="limit"):
        select_event_candidates(np.ones((1, 1)), np.asarray([0]), limit=0, min_separation_ms=0)
    with pytest.raises(ValueError, match="event_count"):
        candidate_orders(0, max_adjacent_swaps=1, max_orders=1)
