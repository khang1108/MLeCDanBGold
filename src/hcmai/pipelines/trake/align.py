"""Exact monotonic DP alignment of TRAKE events over one video's keyframes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores


@dataclass(frozen=True, slots=True)
class TrakePath:
    """One video's best chronological event path, one canonical frame per event."""

    video_id: str
    score: float
    frame_idx: tuple[int, ...]
    frame_ids: tuple[str, ...]


def cluster_starts(scores: np.ndarray, delta: float) -> np.ndarray:
    """Map every frame to the first frame of its cluster, radius ``delta``."""
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
) -> list[TrakePath]:
    """Return the ``paths`` best chronological event paths, one cluster each."""
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
            TrakePath(
                video_id=video.video_id,
                score=float(current[endpoint]),
                frame_idx=tuple(int(video.frame_idx[position]) for position in ordered),
                frame_ids=tuple(str(video.frame_ids[position]) for position in ordered),
            )
        )
    return results
