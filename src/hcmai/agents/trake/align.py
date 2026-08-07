"""Exact monotonic DP alignment of TRAKE events over one video's keyframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shortlist import VideoEventScores


@dataclass(frozen=True, slots=True)
class TrakePath:
    """One video's best strictly chronological event path.

    ``frame_idx`` and ``frame_ids`` hold one canonical frame per event, in
    event order, ready for a ``<video_name>,<frame_1>,...,<frame_N>`` row.
    """

    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]


def align_video(
    video: VideoEventScores, lambda_gap: float = 1e-5, paths: int = 1
) -> list[TrakePath]:
    """Return the best monotonic event-to-frame paths of one shortlisted video.

    Maximizes ``sum_j S[j, t_j] - lambda_gap * (tau_t_N - tau_t_1)`` over
    ``t_1 < ... < t_N`` in ``O(N * M)``: the gap penalties telescope, so a
    running prefix max over ``D[j-1][t'] + lambda * tau_t'`` is enough.

    Args:
        video: One video's ``N x M`` similarities in canonical frame order.
        lambda_gap: Time-gap penalty per millisecond. A calibration knob:
            tune it on a labeled TRAKE validation set.
        paths: How many paths to return, best first, for submission padding.

    Returns:
        Up to ``paths`` paths, best first, empty when ``M < N``. Path ``i``
        ends at the ``i``-th best final frame, so paths may share a prefix.
    """
    scores = np.asarray(video.scores, dtype=np.float64)
    n_events, n_frames = scores.shape
    if n_frames < n_events:
        return []

    weighted_time = lambda_gap * np.asarray(video.timestamps_ms, dtype=np.float64)
    frames = np.arange(n_frames)
    current = scores[0]
    back = np.zeros((n_events, n_frames), dtype=np.int64)
    for event in range(1, n_events):
        shifted = current + weighted_time
        running = np.maximum.accumulate(shifted)
        # Latest position achieving the prefix max: ties keep the shortest gap.
        argmax = np.maximum.accumulate(np.where(shifted == running, frames, 0))
        current = np.full(n_frames, -np.inf)
        current[1:] = scores[event, 1:] - weighted_time[1:] + running[:-1]
        back[event, 1:] = argmax[:-1]

    results = []
    for endpoint in np.argsort(-current)[:paths]:
        if not np.isfinite(current[endpoint]):
            break
        position = int(endpoint)
        path = [position]
        for event in range(n_events - 1, 0, -1):
            position = int(back[event, position])
            path.append(position)
        ordered = tuple(reversed(path))
        results.append(
            TrakePath(
                video_id=video.video_id,
                score=float(current[endpoint]),
                frame_idx=tuple(video.frame_idx[position] for position in ordered),
                frame_ids=tuple(video.frame_ids[position] for position in ordered),
            )
        )
    return results
