"""Video shortlisting and exact per-video rescoring for TRAKE events."""

from __future__ import annotations

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

    Frames are in canonical ``frame_idx`` order, so column ``t`` of
    :attr:`scores` belongs to ``frame_ids[t]``, ``frame_idx[t]`` and
    ``timestamps_ms[t]``. Every value is read from the index mapping, never
    inferred from FPS.
    """

    video_id: str
    frame_ids: tuple[str, ...]
    frame_idx: tuple[int, ...]
    timestamps_ms: np.ndarray
    scores: np.ndarray


def event_video_scores(
    retriever: Any,
    encoder: Any,
    index: Any,
    events: Sequence[str],
    top_k: int = 500,
) -> list[VideoEventScores]:
    """Shortlist videos per event, then rescore every frame of those videos.

    Args:
        retriever: Any ``search(query, top_k)`` retriever (4-modal RRF or dense)
            used only to choose candidate videos.
        encoder: Text encoder producing L2-normalized event vectors.
        index: Loaded :class:`DenseIndex` whose mapping position equals
            ``embedding_index``.
        events: Ordered TRAKE events, already split and translated.
        top_k: Frames kept per event when shortlisting videos.

    Returns:
        One entry per shortlisted video, ordered by ``video_id``.
    """
    if not events:
        raise ValueError("events must not be empty")

    mapping = index.mapping
    timer = Timer()
    frame_ids = {
        candidate.frame_id
        for event in events
        for candidate in retriever.search(event, top_k)
    }
    shortlisted = mapping["video_id"].isin(
        mapping.loc[mapping["frame_id"].isin(frame_ids), "video_id"].unique()
    )
    shortlist_ms = timer.stop()
    timer = Timer()
    scores, positions = index.search(
        encoder.encode_text(list(events)), index.index.ntotal
    )
    full = np.empty_like(scores)
    np.put_along_axis(full, positions, scores, axis=1)
    rescore_ms = timer.stop()

    rows = mapping.loc[shortlisted].sort_values(["video_id", "frame_idx"])
    results = [
        VideoEventScores(
            video_id=str(video_id),
            frame_ids=tuple(group["frame_id"]),
            frame_idx=tuple(int(value) for value in group["frame_idx"]),
            timestamps_ms=group["timestamp_ms"].to_numpy(dtype=np.float64),
            scores=full[:, group["embedding_index"].to_numpy()],
        )
        for video_id, group in rows.groupby("video_id", sort=True)
    ]
    logger.info(
        "TRAKE rescoring events=%d videos=%d frames=%d "
        "shortlist_ms=%.1f rescore_ms=%.1f",
        len(events),
        len(results),
        len(rows),
        shortlist_ms,
        rescore_ms,
    )
    return results
