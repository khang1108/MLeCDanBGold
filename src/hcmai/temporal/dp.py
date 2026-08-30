"""Pure monotonic dynamic programming for ordered event-to-frame alignment.

This module owns only numerical path decoding and per-video path ranking. It
does not retrieve score matrices, resolve canonical frames, or format KIS and
TRAKE responses; those responsibilities remain at higher boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores


@dataclass(frozen=True, slots=True)
class DPPath:
    """One strict chronological event path through a single video's frames."""

    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]


def cluster_starts(scores: np.ndarray, delta: float) -> np.ndarray:
    """Map each frame to its score-cluster start within a video.

    A positive ``delta`` prevents multiple aligned events from landing in one
    near-identical score cluster. The default of zero disables this optional
    diversification constraint.
    """

    columns = np.ascontiguousarray(scores.T)
    starts = np.zeros(len(columns), dtype=np.int64)
    anchor = 0
    for frame in range(1, len(columns)):
        if np.abs(columns[frame] - columns[anchor]).max() > delta:
            anchor = frame
        starts[frame] = anchor
    return starts


def align_video(
    video: VideoEventScores,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> list[DPPath]:
    """Return the highest-scoring strict chronological paths for one video.

    ``VideoEventScores`` columns must already be in canonical frame order. The
    recurrence selects one later column for every successive event and applies
    the configured time-gap penalty without changing any supplied identity.
    """

    scores = np.asarray(video.scores, dtype=np.float64)
    n_events, n_frames = scores.shape
    if n_frames < n_events:
        return []
    if event_power != 1.0:
        scores = np.clip(scores, 0.0, None) ** event_power

    frames = np.arange(n_frames)
    starts = cluster_starts(scores, cluster_delta) if cluster_delta > 0.0 else frames

    if int(np.count_nonzero(starts == frames)) < n_events:
        return []
    
    source = starts - 1
    reachable = source >= 0
    source = source.clip(0)

    weighted_time = lambda_gap * np.asarray(video.timestamps_ms, dtype=np.float64)
    current = scores[0]
    back = np.zeros((n_events, n_frames), dtype=np.int64)
    for event in range(1, n_events):
        shifted = current + weighted_time
        running = np.maximum.accumulate(shifted)
        argmax = np.maximum.accumulate(np.where(shifted == running, frames, 0))
        current = np.where(
            reachable, scores[event] - weighted_time + running[source], -np.inf
        )
        back[event] = np.where(reachable, argmax[source], 0)

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
            DPPath(
                video_id=video.video_id,
                score=float(current[endpoint]),
                frame_idx=tuple(int(video.frame_idx[position]) for position in ordered),
                frame_ids=tuple(str(video.frame_ids[position]) for position in ordered),
            )
        )
    return results


def rank_paths(
    videos: Sequence[VideoEventScores],
    lambda_gap: float = 1e-5,
    max_rows: int = 100,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
) -> list[DPPath]:
    """Rank bounded paths while preferring each video's best row first.

    Diversifying the first ranking level across videos matches the current
    TRAKE behavior and avoids consuming the whole result budget with one
    video's closely related alternatives.
    """

    if not videos:
        return []
    depth = math.ceil(max_rows / len(videos))
    per_video = [
        align_video(video, lambda_gap, depth, event_power, cluster_delta)
        for video in videos
    ]
    rows: list[DPPath] = []
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
