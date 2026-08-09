"""Ranking, video diversification, and TRAKE submission CSV export."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import csv
import math

from hcmai.common.utils.logging import get_logger
from hcmai.retriever.video_scores import VideoEventScores

from .align import TrakePath, align_video

logger = get_logger(__name__)


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


def write_submission(rows: Sequence[TrakePath], output_path: str | Path) -> Path:
    """Write headerless ``<video_name>,<frame_1>,...,<frame_N>`` rows as one CSV."""
    counts = {len(row.frame_idx) for row in rows}
    if len(counts) > 1:
        raise ValueError(f"rows mix event counts: {sorted(counts)}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(
            [row.video_id.removesuffix(".mp4"), *row.frame_idx] for row in rows
        )
    if len(rows) < 100:
        logger.warning(
            "TRAKE submission %s has %d rows, under the 100-row budget",
            path,
            len(rows),
        )
    return path
