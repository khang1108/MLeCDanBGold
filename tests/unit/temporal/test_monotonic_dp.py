"""Smoke tests for exact monotonic DP alignment of TRAKE events."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video


def _video(scores: list[list[float]]) -> VideoEventScores:
    frames = len(scores[0])
    return VideoEventScores(
        video_id="v1",
        frame_ids=tuple(f"v1_{position}" for position in range(frames)),
        frame_idx=tuple(10 * (position + 1) for position in range(frames)),
        timestamps_ms=np.arange(frames, dtype=np.float64) * 1000.0,
        scores=np.array(scores, dtype=np.float32),
    )


def test_path_stays_chronological_when_events_peak_on_one_frame() -> None:
    # Both events peak on frame 1; the path must still increase in time.
    best, second = align_video(
        _video([[0.1, 0.9, 0.2, 0.2], [0.0, 0.8, 0.5, 0.3]]), 0.0, 2
    )
    assert best.frame_ids == ("v1_1", "v1_2")
    assert best.score == pytest.approx(1.4)
    assert second.frame_ids == ("v1_1", "v1_3")


def test_gap_penalty_prefers_the_closer_event_pair() -> None:
    video = _video([[0.5, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.45]])
    assert align_video(video, 0.0)[0].frame_ids == ("v1_0", "v1_3")
    assert align_video(video, 1e-4)[0].frame_ids == ("v1_0", "v1_1")


def test_video_shorter_than_the_event_list_has_no_path() -> None:
    assert align_video(_video([[0.5], [0.4]])) == []


def test_align_video_chooses_the_best_strictly_chronological_path() -> None:
    """Prefer the highest-scoring path that does not reuse a keyframe."""

    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1", "f2", "f3"], dtype=object),
        frame_idx=np.array([0, 1, 2, 3]),
        timestamps_ms=np.array([0, 1_000, 2_000, 3_000]),
        scores=np.array(
            [
                [0.90, 0.20, 0.10, 0.05],
                [0.10, 0.85, 0.30, 0.10],
                [0.05, 0.10, 0.40, 0.95],
            ]
        ),
    )

    [path] = align_video(video, lambda_gap=0.0, paths=1)

    assert path.frame_ids == ("f0", "f1", "f3")


def test_align_video_gap_penalty_can_prefer_a_nearer_frame() -> None:
    """Keep the current time-gap recurrence stable during later relocation."""

    video = VideoEventScores(
        video_id="V01",
        frame_ids=np.array(["f0", "f1", "f2"], dtype=object),
        frame_idx=np.array([0, 1, 2]),
        timestamps_ms=np.array([0, 1_000, 100_000]),
        scores=np.array([[0.90, 0.10, 0.10], [0.10, 0.80, 0.99]]),
    )

    [path] = align_video(video, lambda_gap=1e-5, paths=1)

    assert path.frame_ids == ("f0", "f1")
