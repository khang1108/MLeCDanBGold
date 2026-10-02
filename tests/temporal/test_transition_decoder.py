"""Tests for motion-aware transition graph decoding."""

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video
from hcmai.temporal.transition_decoder import (
    TransitionEdgeMatrix,
    decode_transition_graph,
)


def make_sample_video_scores(
    scores: np.ndarray,
    timestamps_ms: list[int] | None = None,
) -> VideoEventScores:
    n_events, n_frames = scores.shape
    if timestamps_ms is None:
        timestamps_ms = [i * 1000 for i in range(n_frames)]
    return VideoEventScores(
        video_id="video-test-01",
        frame_ids=np.array([f"f_{i}" for i in range(n_frames)]),
        frame_idx=np.array(list(range(n_frames)), dtype=np.int64),
        timestamps_ms=np.array(timestamps_ms, dtype=np.int64),
        scores=scores,
    )


def test_transition_decoder_matches_baseline_when_transitions_none():
    """Verify exact equivalence with frozen baseline DP when no transitions provided."""
    np.random.seed(42)
    scores = np.random.uniform(0.1, 0.9, size=(3, 10))
    video = make_sample_video_scores(scores)

    baseline_paths = align_video(video, lambda_gap=1e-5, paths=3)
    decoded_paths = decode_transition_graph(video, transitions=None, lambda_gap=1e-5, paths=3)

    assert len(decoded_paths) == len(baseline_paths)
    for b_path, d_path in zip(baseline_paths, decoded_paths, strict=True):
        assert d_path.frame_idx == b_path.frame_idx
        assert d_path.frame_ids == b_path.frame_ids
        assert np.isclose(d_path.score, b_path.score)


def test_transition_decoder_matches_baseline_when_transition_weight_zero():
    """Verify exact equivalence when transition weight is zero."""
    np.random.seed(42)
    scores = np.random.uniform(0.1, 0.9, size=(2, 6))
    video = make_sample_video_scores(scores)

    random_edge = np.random.uniform(0.0, 1.0, size=(6, 6))
    baseline_paths = align_video(video, lambda_gap=1e-4, paths=1)
    decoded_paths = decode_transition_graph(
        video,
        transitions=[random_edge],
        transition_weight=0.0,
        lambda_gap=1e-4,
        paths=1,
    )

    assert len(decoded_paths) == len(baseline_paths)
    assert decoded_paths[0].frame_idx == baseline_paths[0].frame_idx
    assert np.isclose(decoded_paths[0].score, baseline_paths[0].score)


def test_transition_matrix_influences_path_selection():
    """Verify that a strong pairwise transition can override a purely unary choice."""
    # 2 events, 4 frames
    # Unary scores favor (0, 3):
    # E0: [0.9, 0.2, 0.1, 0.1] -> favors f0
    # E1: [0.1, 0.1, 0.2, 0.9] -> favors f3
    scores = np.array([
        [0.9, 0.2, 0.1, 0.1],
        [0.1, 0.1, 0.2, 0.9],
    ], dtype=np.float64)
    video = make_sample_video_scores(scores)

    # Baseline selects (0, 3) because 0.9 + 0.9 = 1.8
    baseline_paths = align_video(video, lambda_gap=0.0, paths=1)
    assert baseline_paths[0].frame_idx == (0, 3)

    # Now create a strong transition compatibility between f1 and f2 (+2.0)
    # Total score for (1, 2) becomes 0.2 + 0.2 + 2.0 = 2.4 > 1.8
    edge_matrix = np.zeros((4, 4), dtype=np.float64)
    edge_matrix[1, 2] = 2.0

    decoded_paths = decode_transition_graph(
        video,
        transitions=[edge_matrix],
        transition_weight=1.0,
        lambda_gap=0.0,
        paths=1,
    )

    assert decoded_paths[0].frame_idx == (1, 2)
    assert np.isclose(decoded_paths[0].score, 2.4)


def test_transition_decoder_enforces_chronological_monotonicity():
    """Verify that non-chronological transitions (s >= t) are impossible."""
    scores = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ], dtype=np.float64)
    video = make_sample_video_scores(scores)

    # Even with huge affinity backwards or self-transition, chronological order must hold
    edge_matrix = np.zeros((2, 2), dtype=np.float64)
    edge_matrix[1, 0] = 100.0  # backward transition
    edge_matrix[1, 1] = 100.0  # self transition

    decoded_paths = decode_transition_graph(
        video,
        transitions=[edge_matrix],
        transition_weight=1.0,
        lambda_gap=0.0,
        paths=1,
    )

    assert decoded_paths[0].frame_idx == (0, 1)


def test_transition_edge_matrix_dataclass_support():
    """Verify TransitionEdgeMatrix wrapper object works as expected."""
    scores = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ], dtype=np.float64)
    video = make_sample_video_scores(scores)

    t1 = TransitionEdgeMatrix(
        source_event_index=0,
        target_event_index=1,
        matrix=np.ones((3, 3)) * 0.5,
        weight=1.0,
    )
    t2 = TransitionEdgeMatrix(
        source_event_index=1,
        target_event_index=2,
        matrix=np.ones((3, 3)) * 0.5,
        weight=1.0,
    )

    decoded = decode_transition_graph(
        video,
        transitions=[t1, t2],
        lambda_gap=0.0,
        paths=1,
    )
    assert decoded[0].frame_idx == (0, 1, 2)
    # Score: (0.8 + 0.8 + 0.8) + (0.5 + 0.5) = 3.4
    assert np.isclose(decoded[0].score, 3.4)
