"""Per-video event/frame score matrices for monotonic alignment tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(frozen=True, slots=True)
class VideoEventScores:
    """One video's event-to-frame similarities; column ``t`` is frame ``t`` of every array."""

    video_id: str
    frame_ids: np.ndarray
    frame_idx: np.ndarray
    timestamps_ms: np.ndarray
    scores: np.ndarray


def score_all_videos(
    index: Any,
    query_vectors: np.ndarray,
    chunk_size: int = 65_536,
) -> list[VideoEventScores]:
    """Score every canonical visual-index frame and split scores by video.

    Temporal DP must receive the complete visual corpus so it can compare every
    alignable video. This function deliberately does not perform nearest-
    neighbor shortlisting, reciprocal-rank voting, or metadata filtering.
    """

    positions = np.arange(len(index.frame_ids), dtype=np.int64)
    if len(positions) == 0:
        return []

    scores = index.score_subset(query_vectors, positions, chunk_size)
    video_ids = sorted({str(video_id) for video_id in index.video_ids})
    return [
        VideoEventScores(
            video_id=video_id,
            frame_ids=index.frame_ids[window],
            frame_idx=index.frame_idx[window],
            timestamps_ms=index.timestamps[window],
            scores=scores[:, window],
        )
        for video_id in video_ids
        if len(window := index.video_positions(video_id))
    ]
