"""Video shortlisting and exact per-video rescoring for TRAKE events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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
    frame_ids: tuple[str, ...]
    frame_idx: tuple[int, ...]
    timestamps_ms: np.ndarray
    scores: np.ndarray


def event_video_scores(
    retrieval: Any,
    events: Sequence[str],
    top_k: int = 500,
    max_videos: int = 200,
) -> list[VideoEventScores]:
    """Shortlist videos per event, then rescore every frame of those videos.

    Args:
        retrieval: :class:`RetrievalService`-shaped object providing
            ``search_batch``, ``encode_text_batch`` and ``visual_index``.
        events: Ordered TRAKE events, already split and translated.
        top_k: Frames kept per event when shortlisting videos.
        max_videos: Videos kept for rescoring. A calibration knob: every later
            stage costs ``O(events * frames of these videos)``.

    Returns:
        One entry per shortlisted video, ordered by ``video_id``.
    """
    if not events:
        raise ValueError("events must not be empty")

    index = retrieval.visual_index
    mapping = index.mapping

    timer = Timer()
    ranked = [
        [candidate.frame_id for candidate in result]
        for result in retrieval.search_batch(list(events), top_k)
    ]
    video_of = mapping.loc[
        mapping["frame_id"].isin({frame_id for hits in ranked for frame_id in hits}),
        ["frame_id", "video_id"],
    ].set_index("frame_id")["video_id"].to_dict()

    votes: defaultdict[str, float] = defaultdict(float)
    for hits in ranked:
        seen: set[str] = set()
        for rank, frame_id in enumerate(hits):
            video_id = video_of.get(frame_id)
            if video_id is None or video_id in seen:
                continue
            seen.add(video_id)
            votes[video_id] += 1.0 / (60 + rank)
    shortlisted = mapping["video_id"].isin(
        sorted(votes, key=votes.__getitem__, reverse=True)[:max_videos]
    )
    shortlist_ms = timer.stop()

    rows = mapping.loc[shortlisted].sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    timer.start()
    scores = index.score_subset(
        retrieval.encode_text_batch(list(events), "visual").vectors,
        rows["embedding_index"].to_numpy(),
    )
    rescore_ms = timer.stop()

    results = [
        VideoEventScores(
            video_id=str(video_id),
            frame_ids=tuple(group["frame_id"]),
            frame_idx=tuple(int(value) for value in group["frame_idx"]),
            timestamps_ms=group["timestamp_ms"].to_numpy(dtype=np.float64),
            scores=scores[:, group.index.to_numpy()],
        )
        for video_id, group in rows.groupby("video_id", sort=True)
    ]
    logger.info(
        "TRAKE rescoring events=%d videos=%d/%d frames=%d "
        "shortlist_ms=%.1f rescore_ms=%.1f",
        len(events),
        len(results),
        len(votes),
        len(rows),
        shortlist_ms,
        rescore_ms,
    )
    return results
