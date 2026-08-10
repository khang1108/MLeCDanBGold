"""Per-video event/frame score matrices for monotonic alignment tasks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoEventScores:
    """One video's event-to-frame similarities; column ``t`` is frame ``t`` of every array."""

    video_id: str
    frame_ids: np.ndarray
    frame_idx: np.ndarray
    timestamps_ms: np.ndarray
    scores: np.ndarray


def score_videos(
    index: Any,
    query_vectors: np.ndarray,
    top_k: int = 500,
    max_videos: int = 200,
    rrf_k: int = 60,
    chunk_size: int = 65_536,
) -> list[VideoEventScores]:
    """Shortlist videos by event coverage then RRF vote, and rescore their frames."""
    timer = Timer()
    _, positions = index.search(query_vectors, top_k)

    video_ids = index.video_ids
    votes: defaultdict[str, float] = defaultdict(float)
    coverage: defaultdict[str, int] = defaultdict(int)
    for row in positions:
        seen: set[str] = set()
        for rank, position in enumerate(row):
            if position < 0:
                continue
            video_id = str(video_ids[position])
            if video_id in seen:
                continue
            seen.add(video_id)
            coverage[video_id] += 1
            votes[video_id] += 1.0 / (rrf_k + rank)
    if not votes:
        return []

    ranked = sorted(
        votes,
        key=lambda video_id: (-coverage[video_id], -votes[video_id], video_id),
    )
    shortlist = sorted(ranked[:max_videos])
    windows = [index.video_positions(video_id) for video_id in shortlist]
    scored_positions = np.concatenate(windows)
    shortlist_ms = timer.stop()

    timer.start()
    scores = index.score_subset(query_vectors, scored_positions, chunk_size)
    rescore_ms = timer.stop()

    results = []
    start = 0
    for video_id, window in zip(shortlist, windows):
        stop = start + len(window)
        results.append(
            VideoEventScores(
                video_id=video_id,
                frame_ids=index.frame_ids[window],
                frame_idx=index.frame_idx[window],
                timestamps_ms=index.timestamps[window],
                scores=scores[:, start:stop],
            )
        )
        start = stop
    logger.info(
        "Video rescoring events=%d videos=%d/%d frames=%d "
        "shortlist_ms=%.1f rescore_ms=%.1f",
        len(scores),
        len(results),
        len(votes),
        len(scored_positions),
        shortlist_ms,
        rescore_ms,
    )
    return results
