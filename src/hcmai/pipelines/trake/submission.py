"""Ranking and video diversification for TRAKE candidate paths."""

from __future__ import annotations

from collections.abc import Sequence
import math

from hcmai.retrieval.retriever.video_scores import VideoEventScores

from .align import TrakePath, align_video


def rank_paths(
    videos: Sequence[VideoEventScores],
    lambda_gap: float = 1e-5,
    max_rows: int = 100,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> list[TrakePath]:
    """Rank at most ``max_rows`` aligned paths, one video per row before repeats."""
    if not videos:
        return []
    depth = math.ceil(max_rows / len(videos))
    per_video = [
        align_video(video, lambda_gap, depth, event_power, cluster_delta)
        for video in videos
    ]
    rows: list[TrakePath] = []
    for level in range(depth):
        rows.extend(
            sorted(
                (paths[level] for paths in per_video if len(paths) > level),
                key=lambda path: path.score,
                reverse=True,
            )
        )
        if len(rows) >= max_rows:
            break
    return rows[:max_rows]
