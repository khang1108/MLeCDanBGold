"""Smoke test for exact monotonic DP alignment of TRAKE events."""

from __future__ import annotations

import numpy as np
import pytest

from hcmai.agents.trake import align_video
from hcmai.retriever.video_scores import VideoEventScores


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
    assert best.frame_idx == (20, 30)
    assert best.score == pytest.approx(1.4)
    assert second.frame_idx == (20, 40)


def test_gap_penalty_prefers_the_closer_event_pair() -> None:
    video = _video([[0.5, 0.0, 0.0, 0.0], [0.0, 0.4, 0.0, 0.45]])
    assert align_video(video, 0.0)[0].frame_idx == (10, 40)
    assert align_video(video, 1e-4)[0].frame_idx == (10, 20)


def test_video_shorter_than_the_event_list_has_no_path() -> None:
    assert align_video(_video([[0.5], [0.4]])) == []
