"""Characterization tests for the baseline ordered temporal DP decoder.

These fixtures intentionally pin the current recurrence and ranking behavior
before later Phase A tasks move canonical path materialization out of the
numerical decoder.
"""

from __future__ import annotations

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video, rank_paths


def video_scores(video_id: str, scores: list[list[float]]) -> VideoEventScores:
    """Build a hand-checkable score matrix with canonical frame metadata."""

    matrix = np.asarray(scores, dtype=np.float32)
    n_frames = matrix.shape[1]
    return VideoEventScores(
        video_id=video_id,
        frame_ids=np.asarray([f"{video_id}-f{i}" for i in range(n_frames)]),
        frame_idx=np.arange(n_frames, dtype=np.int64),
        timestamps_ms=np.arange(n_frames, dtype=np.int64) * 1_000,
        scores=matrix,
    )


def test_align_video_is_strictly_increasing_and_full() -> None:
    """Every event must receive one later canonical frame."""

    video = video_scores(
        "v1",
        [
            [9.0, 1.0, 0.0, 0.0],
            [0.0, 8.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 7.0],
        ],
    )

    path = align_video(video, lambda_gap=0.0, paths=1)[0]

    assert path.frame_idx == (0, 1, 3)
    assert len(path.frame_ids) == 3


def test_align_video_returns_no_partial_path_when_video_has_too_few_frames() -> None:
    """Reject videos that cannot assign a distinct frame to every event."""

    video = video_scores("v1", [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    assert align_video(video, lambda_gap=0.0, paths=1) == []


def test_gap_penalty_prefers_shorter_chronological_path() -> None:
    """Preserve the recurrence's configured timestamp-gap tradeoff."""

    video = video_scores(
        "v1",
        [
            [5.0, 4.9, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0],
        ],
    )

    path = align_video(video, lambda_gap=1e-3, paths=1)[0]

    assert path.frame_idx == (1, 2)


def test_one_event_uses_same_decoder() -> None:
    """One event is a one-frame instance of the same DP decoder."""

    video = video_scores("v1", [[0.1, 0.8, 0.2]])

    path = align_video(video, lambda_gap=0.0, paths=1)[0]

    assert path.frame_idx == (1,)


def test_align_video_can_return_multiple_paths_from_same_video() -> None:
    """The per-video decoder may retain several ranked alternatives."""

    video = video_scores("v1", [[5.0, 4.0, 0.0], [0.0, 4.0, 5.0]])

    paths = align_video(video, lambda_gap=0.0, paths=2)

    assert len(paths) == 2
    assert all(path.video_id == "v1" for path in paths)


def test_rank_paths_takes_first_level_across_videos_before_second_level() -> None:
    """Preserve level-wise video diversity before later ranking changes."""

    v1 = video_scores("v1", [[10.0, 9.0, 0.0], [0.0, 9.0, 10.0]])
    v2 = video_scores("v2", [[8.0, 0.0], [0.0, 8.0]])

    rows = rank_paths([v1, v2], lambda_gap=0.0, max_rows=2)

    assert [row.video_id for row in rows] == ["v1", "v2"]
