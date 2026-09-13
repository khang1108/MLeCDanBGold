"""Admissibility-mask tests for ordered temporal dynamic programming."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video


def make_video(scores: list[list[float]]) -> VideoEventScores:
    """Build deterministic canonical metadata for a score matrix."""

    matrix = np.asarray(scores, dtype=np.float32)
    n_frames = matrix.shape[1]
    return VideoEventScores(
        video_id="v1",
        frame_ids=np.asarray([f"f{i}" for i in range(n_frames)]),
        frame_idx=np.arange(n_frames, dtype=np.int64),
        timestamps_ms=np.arange(n_frames, dtype=np.int64) * 1_000,
        scores=matrix,
    )


def test_allowed_mask_blocks_forbidden_states_after_event_power() -> None:
    """Power transforms cannot make forbidden states reachable."""

    original = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.9]], dtype=np.float32)
    video = make_video(original.tolist())
    allowed = np.array([[True, False, False], [False, True, False]])

    paths = align_video(video, lambda_gap=0.0, event_power=2.0, allowed=allowed)

    assert paths[0].frame_ids == ("f0", "f1")
    assert paths[0].score == pytest.approx(1.45)
    np.testing.assert_array_equal(video.scores, original)


def test_mask_does_not_create_clusters() -> None:
    """Clustering is computed from the unmasked transformed scores."""

    video = make_video([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    allowed = np.array([[True, False, False], [False, False, True]])

    assert align_video(video, cluster_delta=0.1, allowed=allowed) == []


def brute_force_best(
    video: VideoEventScores, allowed: np.ndarray, power: float
) -> tuple[float, bool]:
    """Return optimum masked path score and whether any path is feasible."""

    scores = np.clip(np.asarray(video.scores, dtype=float), 0.0, None) ** power
    best = -np.inf
    for columns in combinations(range(scores.shape[1]), scores.shape[0]):
        if not all(allowed[event, column] for event, column in enumerate(columns)):
            continue
        score = sum(scores[event, column] for event, column in enumerate(columns))
        score -= 0.0001 * (
            video.timestamps_ms[columns[-1]] - video.timestamps_ms[columns[0]]
        )
        best = max(best, score)
    return best, np.isfinite(best)


@pytest.mark.parametrize("power", [1.0, 2.0])
def test_masked_dp_matches_brute_force_oracle(power: float) -> None:
    """Masked DP matches exhaustive strict-column feasibility and optimum."""

    rng = np.random.default_rng(12)
    for _ in range(20):
        scores = rng.random((3, 6)).astype(np.float32)
        video = make_video(scores.tolist())
        allowed = scores > 0.35
        expected_score, feasible = brute_force_best(video, allowed, power)
        paths = align_video(
            video, lambda_gap=0.0001, event_power=power, allowed=allowed
        )

        assert bool(paths) == feasible
        if feasible:
            assert paths[0].score == pytest.approx(expected_score, abs=1e-7)


def test_all_true_mask_preserves_exact_existing_output() -> None:
    """An all-true mask is exactly baseline behavior for multiple paths."""

    video = make_video([[5.0, 4.0, 0.0], [0.0, 4.0, 5.0]])
    baseline = align_video(video, lambda_gap=0.0, paths=3)

    assert (
        align_video(
            video, lambda_gap=0.0, paths=3, allowed=np.ones((2, 3), dtype=bool)
        )
        == baseline
    )


def test_unmasked_characterization_fixture() -> None:
    """Pin a nontrivial baseline output independently of all-true parity."""

    video = make_video(
        [[9.0, 1.0, 0.0, 0.0], [0.0, 8.0, 1.0, 0.0], [0.0, 0.0, 1.0, 7.0]]
    )

    paths = align_video(video, lambda_gap=0.0, paths=2)

    assert [(path.frame_idx, path.score) for path in paths] == [
        ((0, 1, 3), 24.0),
        ((0, 1, 2), 18.0),
    ]


@pytest.mark.parametrize(
    "scores, allowed, expected",
    [
        ([[0.1, 0.8]], [[False, True]], (1,)),
        (
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[True, True], [True, True], [True, True]],
            None,
        ),
        ([[1.0, 0.0], [0.0, 1.0]], [[False, False], [True, True]], None),
        (
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[True, False, False], [False, False, True]],
            (0, 2),
        ),
    ],
)
def test_mask_edge_shapes_and_order(scores, allowed, expected) -> None:
    """Cover one event, too few frames, empty rows, and feasible order."""

    paths = align_video(
        make_video(scores), lambda_gap=0.0, allowed=np.asarray(allowed, dtype=bool)
    )

    if expected is None:
        assert paths == []
    else:
        assert paths[0].frame_idx == expected


def test_reversed_admissible_order_has_no_strict_path() -> None:
    """Reject masks whose only event choices run backward in time."""

    video = make_video([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    allowed = np.array([[False, False, True], [True, False, False]])

    assert align_video(video, lambda_gap=0.0, allowed=allowed) == []


@pytest.mark.parametrize(
    "allowed",
    [
        np.ones((1, 3), dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.ones((2, 3), dtype=np.int8),
    ],
)
def test_invalid_allowed_mask_raises_value_error(allowed: np.ndarray) -> None:
    """Reject masks with wrong shape or dtype using one stable error."""

    video = make_video([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    with pytest.raises(
        ValueError, match="allowed must be a boolean event-by-frame mask"
    ):
        align_video(video, allowed=allowed)


def test_invalid_allowed_is_validated_before_early_return() -> None:
    """Invalid masks still fail when frame count cannot satisfy event count."""

    video = make_video([[1.0], [1.0]])

    with pytest.raises(
        ValueError, match="allowed must be a boolean event-by-frame mask"
    ):
        align_video(video, allowed=np.ones((1, 1), dtype=bool))
