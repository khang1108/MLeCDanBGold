"""Tests for event candidate selection in motion-aware graph decoding."""

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.transition_decoder import (
    EventCandidateLayer,
    select_event_candidates,
)


def make_sample_video(scores: np.ndarray) -> VideoEventScores:
    n_events, n_frames = scores.shape
    return VideoEventScores(
        video_id="video-test-candidates",
        frame_ids=np.array([f"f_{i}" for i in range(n_frames)]),
        frame_idx=np.array(list(range(n_frames)), dtype=np.int64),
        timestamps_ms=np.array([i * 1000 for i in range(n_frames)], dtype=np.int64),
        scores=scores,
    )


def test_invalid_k_raises_value_error():
    scores = np.array([[1.0, 2.0], [2.0, 3.0]])
    video = make_sample_video(scores)
    with pytest.raises(ValueError, match="candidate_k must be positive"):
        select_event_candidates(video, candidate_k=0)

    with pytest.raises(ValueError, match="candidate_k must be positive"):
        select_event_candidates(video, candidate_k=-5)


def test_first_frame_regression():
    """Verify that event 0 candidates can include frame 0."""
    scores = np.array([
        [10.0, 1.0, 0.0],
        [0.0, 9.0, 8.0],
    ])
    video = make_sample_video(scores)
    layers = select_event_candidates(video, candidate_k=2)

    assert len(layers) == 2
    # Event 0: top-2 frames by score [10, 1, 0] are frames 0 and 1
    assert 0 in layers[0].frame_indices
    np.testing.assert_array_equal(layers[0].frame_indices, np.array([0, 1]))
    np.testing.assert_allclose(layers[0].scores, np.array([10.0, 1.0]))

    # Event 1: top-2 frames by score [0, 9, 8] are frames 1 and 2
    np.testing.assert_array_equal(layers[1].frame_indices, np.array([1, 2]))
    np.testing.assert_allclose(layers[1].scores, np.array([9.0, 8.0]))


def test_large_k_does_not_crash():
    scores = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    video = make_sample_video(scores)
    layers = select_event_candidates(video, candidate_k=100)

    assert len(layers) == 2
    assert len(layers[0].frame_indices) == 3
    # Event 1 can only be at frames that have a predecessor (frames 1 and 2)
    assert len(layers[1].frame_indices) == 2
