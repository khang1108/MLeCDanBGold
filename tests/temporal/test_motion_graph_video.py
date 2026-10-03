"""Tests for end-to-end decode_motion_graph_video (SP-08).

Verifies Task 5.1:
- End-to-end orchestration: select candidates -> fetch candidate embeddings -> score transitions -> decode lattice.
- Transition evidence correctly flips alignment when unary evidence is ambiguous or reversed.
- Transition weight 0 matches static candidate decoding.
"""

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video
from hcmai.temporal.transition_decoder import (
    FrameEmbeddingAccessor,
    decode_motion_graph_video,
)


def _make_motion_test_video() -> tuple[VideoEventScores, np.ndarray, np.ndarray]:
    """Create a video with 2 candidate paths:

    Path A (0 -> 2): moderate unary (0.6 + 0.6 = 1.2), but correct forward transition.
    Path B (1 -> 3): high unary (0.7 + 0.7 = 1.4), but reversed transition.
    """
    scores = np.array([
        [0.6, 0.7, 0.1, 0.1],  # Event 0: frame 1 looks higher than frame 0
        [0.1, 0.1, 0.6, 0.7],  # Event 1: frame 3 looks higher than frame 2
    ])

    video = VideoEventScores(
        video_id="video-motion-test",
        frame_ids=np.array(["f0", "f1", "f2", "f3"]),
        frame_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        scores=scores,
    )

    # 2D feature space: dim 0 = sit, dim 1 = stand
    # Query transition: sit [1, 0] -> stand [0, 1]
    event_embs = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    # Frame embeddings:
    # Frame 0: sit [1, 0]
    # Frame 1: stand [0, 1]
    # Frame 2: stand [0, 1]  (0 -> 2 is sit -> stand: FORWARD transition!)
    # Frame 3: sit [1, 0]    (1 -> 3 is stand -> sit: REVERSED transition!)
    frame_embs = np.array([
        [1.0, 0.0],  # f0: sit
        [0.0, 1.0],  # f1: stand
        [0.0, 1.0],  # f2: stand
        [1.0, 0.0],  # f3: sit
    ], dtype=np.float32)

    return video, frame_embs, event_embs


def test_motion_graph_flips_reversed_unary_path():
    """Verify that query-conditioned transition scoring overcomes deceptive unary scores."""
    video, frame_embs, event_embs = _make_motion_test_video()

    # 1. Without transition edges (transition_weight = 0.0):
    # Deceptive unary scores pick Path B (1 -> 3) with score 1.4
    static_paths = decode_motion_graph_video(
        video,
        frame_embeddings=frame_embs,
        event_embeddings=event_embs,
        candidate_k=4,
        transition_weight=0.0,
        lambda_gap=0.0,
    )
    assert len(static_paths) == 1
    assert static_paths[0].frame_idx == (1, 3)
    assert np.isclose(static_paths[0].score, 1.4)

    # 2. With motion-aware transition edges (transition_weight = 1.0):
    # Path A (0 -> 2): unary = 1.2, transition psi(0, 2) = +0.5 -> total = 1.7
    # Path B (1 -> 3): unary = 1.4, transition psi(1, 3) = -0.5 -> total = 0.9
    # Path A wins!
    motion_paths = decode_motion_graph_video(
        video,
        frame_embeddings=frame_embs,
        event_embeddings=event_embs,
        candidate_k=4,
        transition_weight=1.0,
        lambda_gap=0.0,
    )
    assert len(motion_paths) == 1
    assert motion_paths[0].frame_idx == (0, 2)
    assert np.isclose(motion_paths[0].score, 1.7)


def test_motion_graph_video_matches_baseline_when_weight_zero():
    """Verify decode_motion_graph_video matches baseline align_video when weight=0.0."""
    video, frame_embs, event_embs = _make_motion_test_video()

    baseline_paths = align_video(video, lambda_gap=1e-5, paths=2)
    motion_paths = decode_motion_graph_video(
        video,
        frame_embeddings=frame_embs,
        event_embeddings=event_embs,
        candidate_k=4,
        transition_weight=0.0,
        lambda_gap=1e-5,
        paths=2,
    )

    assert len(baseline_paths) == len(motion_paths)
    for b_p, m_p in zip(baseline_paths, motion_paths, strict=True):
        assert b_p.frame_idx == m_p.frame_idx
        assert np.isclose(b_p.score, m_p.score)


def test_motion_graph_video_with_frame_embedding_accessor():
    """Verify decode_motion_graph_video works with FrameEmbeddingAccessor."""
    video, frame_embs, event_embs = _make_motion_test_video()
    accessor = FrameEmbeddingAccessor({"video-motion-test": frame_embs})

    paths = decode_motion_graph_video(
        video,
        frame_embeddings=accessor,
        event_embeddings=event_embs,
        candidate_k=4,
        transition_weight=1.0,
        lambda_gap=0.0,
    )

    assert len(paths) == 1
    assert paths[0].frame_idx == (0, 2)
    assert np.isclose(paths[0].score, 1.7)
