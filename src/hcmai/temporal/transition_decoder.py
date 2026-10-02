"""Motion-Aware Graph Decoding for ordered event-to-frame alignment.

Extends monotonic temporal dynamic programming with pairwise candidate transition
compatibility matrices :math:`\\psi(s, t; E_{i-1}, E_i)` representing motion consistency,
action flow continuity, or cross-event transition affinity.

When transition edge weights are absent or zero, this recurrence is mathematically
and numerically equivalent to the frozen baseline dynamic programming in
``hcmai.temporal.dp``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import (
    AlignedPath,
    ConditionedDPPath,
    DPPath,
    _prepare_dp_inputs,
    align_video,
)


@dataclass(frozen=True, slots=True)
class TransitionEdgeMatrix:
    """Pairwise transition compatibility scores between frames for adjacent events."""

    source_event_index: int
    target_event_index: int
    matrix: np.ndarray
    weight: float = 1.0


def decode_transition_graph(
    video: VideoEventScores,
    transitions: Sequence[np.ndarray | TransitionEdgeMatrix] | None = None,
    transition_weight: float = 1.0,
    lambda_gap: float = 1e-5,
    paths: int = 1,
    event_power: float = 1.0,
    cluster_delta: float = 0.0,
    min_separation_ms: int = 0,
    *,
    allowed: np.ndarray | None = None,
) -> list[DPPath]:
    """Decode optimal chronological paths through video frames using motion-aware transition edges.

    Recurrence:
    .. math::

        DP_i(t) = U_i(t) + \\max_{s < t} [ DP_{i-1}(s) + \\psi_{i-1, i}(s, t) - \\lambda (t - s) ]

    Args:
        video: VideoEventScores holding per-event frame scores and timestamps.
        transitions: Optional sequence of transition compatibility matrices (length n_events - 1).
        transition_weight: Global scaling weight for transition edge scores.
        lambda_gap: Linear time-gap penalty weight.
        paths: Maximum number of ranked non-overlapping paths to return.
        event_power: Power scaling applied to positive unary scores.
        cluster_delta: Score drift threshold for frame clustering.
        min_separation_ms: Minimum timestamp separation between alternative paths.
        allowed: Optional boolean admissibility mask of shape (n_events, n_frames).

    Returns:
        Ranked list of DPPath instances.
    """
    # Fast-path: when no transition matrices or zero weight, delegate to baseline
    if (transitions is None or transition_weight == 0.0) and cluster_delta == 0.0:
        return align_video(
            video,
            lambda_gap=lambda_gap,
            paths=paths,
            event_power=event_power,
            cluster_delta=cluster_delta,
            min_separation_ms=min_separation_ms,
            allowed=allowed,
        )

    prep = _prepare_dp_inputs(video, allowed, event_power, cluster_delta)
    if prep is None:
        return []
    scores, frames, starts, source, reachable = prep
    n_events, n_frames = scores.shape

    weighted_time = lambda_gap * np.asarray(video.timestamps_ms, dtype=np.float64)
    current = scores[0].copy()
    back = np.zeros((n_events, n_frames), dtype=np.int64)

    # Pre-build predecessor validity mask: valid[s, t] is True iff s <= source[t] and s >= 0
    # source is shape (n_frames,); source[t] is the highest valid predecessor frame index
    s_indices = np.arange(n_frames)[:, None]  # shape (n_frames, 1)
    t_sources = source[None, :]               # shape (1, n_frames)
    t_reachable = reachable[None, :]          # shape (1, n_frames)
    valid_predecessors = (s_indices <= t_sources) & t_reachable

    for event in range(1, n_events):
        edge_idx = event - 1
        edge_mat: np.ndarray | None = None
        if transitions is not None and edge_idx < len(transitions):
            t_spec = transitions[edge_idx]
            if isinstance(t_spec, TransitionEdgeMatrix):
                edge_mat = np.asarray(t_spec.matrix, dtype=np.float64) * (t_spec.weight * transition_weight)
            elif t_spec is not None:
                edge_mat = np.asarray(t_spec, dtype=np.float64) * transition_weight

        shifted_prev = current + weighted_time

        if edge_mat is None or np.all(edge_mat == 0.0):
            # No edge weight: use O(n) prefix maximum
            running = np.maximum.accumulate(shifted_prev)
            argmax = np.maximum.accumulate(np.where(shifted_prev == running, frames, 0))
            current = np.where(
                reachable, scores[event] - weighted_time + running[source], -np.inf
            )
            back[event] = np.where(reachable, argmax[source], 0)
        else:
            # Motion-aware edge weight: candidate compatibility matrix
            cand_scores = shifted_prev[:, None] + edge_mat
            # Mask out invalid predecessors
            masked_cand = np.where(
                valid_predecessors & np.isfinite(shifted_prev)[:, None],
                cand_scores,
                -np.inf,
            )
            best_s = np.argmax(masked_cand, axis=0)
            best_val = np.max(masked_cand, axis=0)

            current = np.where(
                reachable & np.isfinite(best_val),
                scores[event] - weighted_time + best_val,
                -np.inf,
            )
            back[event] = np.where(reachable, best_s, 0)

    timestamps = np.asarray(video.timestamps_ms, dtype=np.int64)
    results: list[DPPath] = []
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

        for event in range(n_events - 1, 0, -1):
            position = int(back[event, position])
            path.append(position)
        ordered = tuple(reversed(path))
        results.append(
            DPPath(
                video_id=video.video_id,
                score=float(current[endpoint]),
                frame_idx=tuple(int(video.frame_idx[pos]) for pos in ordered),
                frame_ids=tuple(str(video.frame_ids[pos]) for pos in ordered),
            )
        )
    return results


__all__ = [
    "AlignedPath",
    "ConditionedDPPath",
    "DPPath",
    "TransitionEdgeMatrix",
    "decode_transition_graph",
]
