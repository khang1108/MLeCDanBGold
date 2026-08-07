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
    """One shortlisted video's event-to-frame similarities.

    Column ``t`` of :attr:`scores` belongs to ``frame_ids[t]``,
    ``frame_idx[t]`` and ``timestamps_ms[t]``, all read from the index mapping
    in canonical order and never inferred from FPS.
    """

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
    """Shortlist videos by event coverage then RRF vote, and rescore their frames.

    Args:
        index: :class:`DenseIndex`-shaped object providing ``search``,
            ``score_subset``, ``video_positions`` and the position-indexed
            ``video_ids``/``frame_ids``/``frame_idx``/``timestamps_ms`` arrays.
        query_vectors: One L2-normalized row per ordered event.
        top_k: Frames kept per event when shortlisting videos.
        max_videos: Videos kept for rescoring, filled from full coverage down.
        rrf_k: RRF constant damping the head of each event's ranking.
        chunk_size: Vectors reconstructed at a time, bounding peak memory.

    Returns:
        One entry per shortlisted video, ordered by ``video_id``.
    """
    timer = Timer()
    _, positions = index.search(query_vectors, top_k)

    video_ids = index.video_ids
    votes: defaultdict[str, float] = defaultdict(float)
    coverage: defaultdict[str, int] = defaultdict(int)
    for row in positions:
        seen: set[str] = set()
        for rank, position in enumerate(row):
            if position < 0:  # FAISS pads short result rows with -1.
                continue
            video_id = str(video_ids[position])
            if video_id in seen:
                continue
            seen.add(video_id)
            coverage[video_id] += 1
            votes[video_id] += 1.0 / (rrf_k + rank)
    if not votes:
        return []

    # Coverage first: TRAKE needs every event inside one video, so evidence for
    # all of them beats one very strong match however high it ranks. RRF breaks
    # ties within a coverage tier, video_id keeps the order deterministic.
    postings = index.video_positions
    ranked = sorted(
        votes,
        key=lambda video_id: (-coverage[video_id], -votes[video_id], video_id),
    )
    shortlist = sorted(ranked[:max_videos])
    windows = [postings[video_id] for video_id in shortlist]
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
                timestamps_ms=index.timestamps_ms[window],
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
