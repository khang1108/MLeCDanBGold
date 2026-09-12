"""Exact replay gate for all baseline methods on fixed score matrices."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.contracts import BaselinePath, MethodName, MethodOptions
from baseline.methods import create_method
from hcmai.retrieval.retriever.video_scores import VideoEventScores


def _video(video_id: str, scores: list[list[float]]) -> VideoEventScores:
    matrix = np.asarray(scores, dtype=np.float32)
    frame_count = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray(
            [f"{video_id}-f{position}" for position in range(frame_count)]
        ),
        frame_idx=np.arange(10, 10 + frame_count, dtype=np.int64),
        timestamps_ms=np.arange(frame_count, dtype=np.int64) * 1_000,
        scores=matrix,
    )


# Alpha's best second event precedes its best first event. The strict sparse
# lattice rejects that path, while bounded order recovers the inversion.
_ALPHA_GLOBAL = _video("alpha", [[0.0, 0.0, 9.0, 0.0]])
_BETA_GLOBAL = _video("beta", [[6.0, 0.0, 0.0, 0.0]])
_ALPHA_EVENTS = _video(
    "alpha",
    [
        [0.0, 0.0, 9.0, 0.0],
        [8.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 7.0],
    ],
)
_BETA_EVENTS = _video(
    "beta",
    [
        [6.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ],
)
_OPTIONS = MethodOptions(
    lambda_gap=0.0,
    event_power=1.0,
    cluster_delta=0.0,
    candidates_per_event=1,
    candidate_min_separation_ms=0,
    max_adjacent_swaps=1,
    max_order_candidates=64,
    swap_penalty=0.0,
)


@pytest.mark.parametrize(
    ("method", "videos", "expected"),
    [
        (
            "global_query",
            [_ALPHA_GLOBAL, _BETA_GLOBAL],
            [
                BaselinePath("alpha", 9.0, ("alpha-f2",), (12,), (2_000,), None),
                BaselinePath("beta", 6.0, ("beta-f0",), (10,), (0,), None),
            ],
        ),
        (
            "independent_events",
            [_ALPHA_EVENTS, _BETA_EVENTS],
            [
                BaselinePath(
                    "alpha",
                    8.0,
                    ("alpha-f2", "alpha-f0", "alpha-f3"),
                    (12, 10, 13),
                    (2_000, 0, 3_000),
                    None,
                ),
                BaselinePath(
                    "beta",
                    5.0,
                    ("beta-f0", "beta-f1", "beta-f3"),
                    (10, 11, 13),
                    (0, 1_000, 3_000),
                    None,
                ),
            ],
        ),
        (
            "dante_style_unary_dp",
            [_ALPHA_EVENTS, _BETA_EVENTS],
            [
                BaselinePath(
                    "beta",
                    15.0,
                    ("beta-f0", "beta-f1", "beta-f3"),
                    (10, 11, 13),
                    (0, 1_000, 3_000),
                    (0, 1, 2),
                ),
                BaselinePath(
                    "alpha",
                    7.0,
                    ("alpha-f1", "alpha-f2", "alpha-f3"),
                    (11, 12, 13),
                    (1_000, 2_000, 3_000),
                    (0, 1, 2),
                ),
            ],
        ),
        (
            "unary_candidate_lattice",
            [_ALPHA_EVENTS, _BETA_EVENTS],
            [
                BaselinePath(
                    "beta",
                    15.0,
                    ("beta-f0", "beta-f1", "beta-f3"),
                    (10, 11, 13),
                    (0, 1_000, 3_000),
                    (0, 1, 2),
                )
            ],
        ),
        (
            "unary_bounded_order",
            [_ALPHA_EVENTS, _BETA_EVENTS],
            [
                BaselinePath(
                    "alpha",
                    24.0,
                    ("alpha-f2", "alpha-f0", "alpha-f3"),
                    (12, 10, 13),
                    (2_000, 0, 3_000),
                    (1, 0, 2),
                ),
                BaselinePath(
                    "beta",
                    15.0,
                    ("beta-f0", "beta-f1", "beta-f3"),
                    (10, 11, 13),
                    (0, 1_000, 3_000),
                    (0, 1, 2),
                ),
            ],
        ),
    ],
)
def test_method_replay_preserves_exact_identity_order_and_score(
    method: MethodName,
    videos: list[VideoEventScores],
    expected: list[BaselinePath],
) -> None:
    assert create_method(method, _OPTIONS).rank(videos, top_k=2) == expected
