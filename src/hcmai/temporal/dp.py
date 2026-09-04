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


@dataclass(frozen=True, slots=True)
class AlignedPath:
    """A canonical temporal path materialized from one decoded DP row."""

    video_id: str
    score: float
    frame_ids: tuple[str, ...]
    frame_idxs: tuple[int, ...]
    timestamps_ms: tuple[int, ...]


def cluster_starts(scores: np.ndarray, delta: float) -> np.ndarray:
    """Map each frame to its score-cluster start within a video.

    A positive ``delta`` prevents multiple aligned events from landing in one
    near-identical score cluster. The default of zero disables this optional
    diversification constraint.

    Args:
        scores: np.ndarray
            A [event, frame] array of shape (n_events, n_frames) containing the event-to-frame scores.
        delta: float
            The minimum score difference required to start a new cluster.

    Returns:
        np.ndarray
            An array of length n_frames where each element indicates the start of the score cluster for that frame.
    """

    # ``scores`` is indexed as [event, frame], but clustering compares a
    # complete score profile per frame. Transposing makes ``columns[t]`` the
    # vector [score(event_0, t), score(event_1, t), ...]. A contiguous layout
    # avoids a strided view while the loop repeatedly reads these vectors.
    columns = np.ascontiguousarray(scores.T)

    # ``starts[t]`` will hold the first frame position of the cluster that
    # contains frame ``t``. Frame zero always begins the first cluster.
    starts = np.zeros(len(columns), dtype=np.int64)

    anchor = 0
    for frame in range(1, len(columns)):
        # Compare to the cluster anchor rather than the immediately preceding
        # frame. This bounds the whole cluster's drift from its first profile.
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
    min_separation_ms: int = 0,
) -> list[DPPath]:
    """Return the highest-scoring strict chronological paths for one video.

    ``VideoEventScores`` columns must already be in canonical frame order. The
    recurrence selects one later column for every successive event and applies
    the configured time-gap penalty without changing any supplied identity.

    A positive ``min_separation_ms`` suppresses alternatives whose final event
    lands near an already accepted path, so extra rows are distinct moments
    rather than neighbouring frames of the same one.
    """

    scores = np.asarray(video.scores, dtype=np.float64)
    n_events, n_frames = scores.shape

    if n_frames < n_events:
        return []
    if event_power != 1.0:
        scores = np.clip(scores, 0.0, None) ** event_power

    frames = np.arange(n_frames)

    # With clustering disabled, every frame starts its own admissible region.
    # With it enabled, a later event may only enter after the prior region.
    starts = (
        cluster_starts(scores, cluster_delta)
        if cluster_delta > 0.0
        else frames
    )

    if int(np.count_nonzero(starts == frames)) < n_events:
        return []

    # Event ``e`` at a frame in a cluster beginning at ``s`` can only inherit
    # from a predecessor at or before ``s - 1``. This enforces strict frame
    # order normally and also prevents reuse within a clustered score region.
    source = starts - 1
    reachable = source >= 0
    source = source.clip(0)

    # The recurrence is score(previous) + score(current) - lambda_gap *
    # (timestamp(current) - timestamp(previous)). Rewriting it below as
    # ``previous + weighted_time`` then ``current - weighted_time`` permits a
    # prefix maximum instead of an O(n_frames^2) predecessor search.
    weighted_time = lambda_gap * np.asarray(video.timestamps_ms, dtype=np.float64)
    current = scores[0]
    back = np.zeros((n_events, n_frames), dtype=np.int64)
    
    for event in range(1, n_events):
        shifted = current + weighted_time

        # At each position t, ``running[t]`` is the best predecessor score
        # among frames 0..t. ``argmax`` records the matching frame for later
        # reconstruction; ties deliberately use the latest matching frame.
        running = np.maximum.accumulate(shifted)
        argmax = np.maximum.accumulate(np.where(shifted == running, frames, 0))

        # ``source`` limits each endpoint to valid predecessors. Frame/cluster
        # starts with no earlier predecessor are explicitly unreachable.
        current = np.where(
            reachable, scores[event] - weighted_time + running[source], -np.inf
        )
        back[event] = np.where(reachable, argmax[source], 0)

    timestamps = np.asarray(video.timestamps_ms, dtype=np.int64)
    results = []
    accepted: list[int] = []
    for endpoint in np.argsort(-current):
        if len(results) >= paths:
            break
        if not np.isfinite(current[endpoint]):
            break
        position = int(endpoint)
        if any(
            abs(int(timestamps[position]) - taken) < min_separation_ms
            for taken in accepted
        ):
            continue
        accepted.append(int(timestamps[position]))
        path = [position]

        # Follow one stored predecessor for each earlier event, then reverse
        # because decoding begins from the final event's endpoint.
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
    paths_per_video: int = 1,
    path_min_separation_ms: int = 0,
) -> list[DPPath]:
    """Rank bounded paths while preferring each video's best row first.

    Diversifying the first ranking level across videos matches the current
    TRAKE behavior and avoids consuming the whole result budget with one
    video's closely related alternatives. ``paths_per_video`` above one lets a
    video offer further separated moments once every video has placed its best.
    """

    if not videos:
        return []

    # Ask every video for enough alternatives that taking rows level-by-level
    # can fill ``max_rows`` even when some videos have no valid path.
    depth = max(paths_per_video, math.ceil(max_rows / len(videos)))
    per_video = [
        align_video(
            video,
            lambda_gap,
            depth,
            event_power,
            cluster_delta,
            path_min_separation_ms,
        )
        for video in videos
    ]
    if paths_per_video > 1:
        # Level-wise ranking reserves level zero for one row per video, so with
        # far more videos than ``max_rows`` a second moment can never surface.
        # Ranking every retained alternative by score is what lets one video
        # offer an alternative moment, bounded by ``paths_per_video``.
        return sorted(
            (path for paths in per_video for path in paths),
            key=lambda path: path.score,
            reverse=True,
        )[:max_rows]

    rows: list[DPPath] = []
    for level in range(depth):
        # Level zero contains each video's best path, level one its second
        # best, etc. Sorting only inside a level preserves this diversity rule.
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
