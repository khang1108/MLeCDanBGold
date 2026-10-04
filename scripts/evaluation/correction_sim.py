"""Simulate a user who repairs one decoded event at a time on a frozen score matrix.

The oracle knows per-event ground-truth intervals. Nothing here touches the corpus:
each action only edits an admissibility mask and re-decodes the same VideoEventScores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hcmai.event_trail.decoding.decoder import TemporalConstraintDecoder
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video, align_video_conditioned


# rejection_cell is pure timestamp arithmetic; it never touches the temporal service.
_REJECTION = TemporalConstraintDecoder(temporal=None)  # type: ignore[arg-type]


def event_margins(
    video: VideoEventScores,
    *,
    allowed: np.ndarray | None = None,
    lambda_gap: float = 1e-5,
    min_separation_ms: int = 5_000,
) -> np.ndarray:
    """Return, per event, the path score lost by moving it to its best distinct alternative.

    A small margin means another clearly separated moment fits the whole sequence
    almost as well. ``inf`` means no separated alternative exists; ``nan`` means no valid path.
    """
    margins = np.full(video.scores.shape[0], np.nan)
    for event in range(video.scores.shape[0]):
        paths = align_video_conditioned(
            video, event, allowed=allowed, lambda_gap=lambda_gap,
            max_paths=2, min_separation_ms=min_separation_ms,
        )
        if len(paths) == 2:
            margins[event] = paths[0].score - paths[1].score
        elif paths:
            margins[event] = np.inf
    return margins


@dataclass(frozen=True)
class SimConfig:
    """One experimental condition. Defaults mirror shipped EventTrail settings."""

    proposals: str  # "local" (top frames by S[e,f]) | "path" (max-marginal)
    order: str  # "sequential" | "random" | "margin"
    k: int = 4
    separation_ms: int = 5_000
    tolerance_ms: int = 0  # widen every ground-truth interval by this much on both sides
    budget: int = 5
    propagate: bool = True  # False: an action re-places only the touched event
    lambda_gap: float = 1e-5
    seed: int = 0


def _decode(video: VideoEventScores, allowed: np.ndarray, cfg: SimConfig) -> list[int] | None:
    """Return matrix positions of the best path under the mask, or None."""
    paths = align_video(video, cfg.lambda_gap, 1, allowed=allowed)
    if not paths:
        return None
    position = {str(fid): i for i, fid in enumerate(video.frame_ids)}
    return [position[fid] for fid in paths[0].frame_ids]


def _repair_one(video: VideoEventScores, allowed: np.ndarray, path: list[int], event: int, cfg: SimConfig) -> list[int] | None:
    """No-propagation baseline: re-decode only ``event`` with every other event frozen."""
    if not allowed[event].any():
        return None
    pinned = np.zeros_like(allowed)
    pinned[event] = allowed[event]
    for other, pos in enumerate(path):
        if other != event:
            pinned[other, pos] = True
    repaired = _decode(video, pinned, cfg)
    if repaired is not None:
        return repaired

    # The edit breaks order with a frozen neighbour; a local editor would apply it anyway.
    pos = int(np.argmax(np.where(allowed[event], video.scores[event], -np.inf)))
    return path[:event] + [pos] + path[event + 1:]


def _inside(video: VideoEventScores, pos: int, interval: tuple[int, int]) -> bool:
    """Whether a matrix position's timestamp falls in a closed ms interval."""
    t = int(video.timestamps_ms[pos])
    return interval[0] <= t <= interval[1]


def _local_proposals(video: VideoEventScores, allowed: np.ndarray, event: int, cfg: SimConfig) -> list[int]:
    """Top-k admissible frames by the event's own score, temporally separated."""
    ranked = np.argsort(-np.where(allowed[event], video.scores[event], -np.inf))
    picked: list[int] = []
    for pos in ranked:
        if not allowed[event, pos] or len(picked) == cfg.k:
            break
        t = int(video.timestamps_ms[pos])
        if all(abs(t - int(video.timestamps_ms[p])) >= cfg.separation_ms for p in picked):
            picked.append(int(pos))
    return picked


def _path_proposals(video: VideoEventScores, allowed: np.ndarray, event: int, cfg: SimConfig) -> list[int]:
    """Top-k focus positions ranked by the best complete path through them."""
    return [
        cp.focus_frame_position
        for cp in align_video_conditioned(
            video,
            event,
            allowed=allowed,
            lambda_gap=cfg.lambda_gap,
            max_paths=cfg.k,
            min_separation_ms=cfg.separation_ms,
        )
    ]


def simulate(video: VideoEventScores, gt: list[tuple[int, int]], cfg: SimConfig) -> dict:
    """Count actions until every event sits inside its interval or the budget runs out."""
    gt = [(start - cfg.tolerance_ms, end + cfg.tolerance_ms) for start, end in gt]
    n_events = video.scores.shape[0]
    allowed = np.ones(video.scores.shape, dtype=bool)
    anchored: set[int] = set()
    rng = np.random.default_rng(cfg.seed)
    propose = _path_proposals if cfg.proposals == "path" else _local_proposals

    path = _decode(video, allowed, cfg)
    correct_at_start = path is not None and all(_inside(video, path[e], gt[e]) for e in range(n_events))
    actions = 0
    auto_fixed = auto_broken = 0

    while path is not None and actions < cfg.budget:
        wrong = {e for e in range(n_events) if not _inside(video, path[e], gt[e])}
        candidates = [e for e in range(n_events) if e not in anchored]
        if not wrong or not candidates:
            break

        # The order strategy only sees what a real system sees; it never peeks at ``wrong``.
        if cfg.order == "sequential":
            event = candidates[0]
        elif cfg.order == "random":
            event = int(rng.choice(candidates))
        else:
            margins = event_margins(
                video,
                allowed=allowed,
                lambda_gap=cfg.lambda_gap,
                min_separation_ms=cfg.separation_ms,
            )
            event = min(candidates, key=lambda e: margins[e])
        actions += 1

        if event not in wrong:
            # Keep: the user confirms the current placement.
            allowed[event] = False
            allowed[event, path[event]] = True
            anchored.add(event)
        else:
            hit = next((p for p in propose(video, allowed, event, cfg) if _inside(video, p, gt[event])), None)
            if hit is not None:
                # Use: the user picks the correct alternative.
                allowed[event] = False
                allowed[event, hit] = True
                anchored.add(event)
            else:
                # Reject the same midpoint-split cell that SHI's EventTrail rejects.
                left, right = _REJECTION.rejection_cell(video, str(video.frame_ids[path[event]]))
                allowed[event] &= (video.timestamps_ms < left) | (video.timestamps_ms > right)

        before = path
        path = _decode(video, allowed, cfg) if cfg.propagate else _repair_one(video, allowed, path, event, cfg)
        if path is not None:
            for other in range(n_events):
                if other == event:
                    continue
                was_right = _inside(video, before[other], gt[other])
                now_right = _inside(video, path[other], gt[other])
                auto_fixed += now_right and not was_right
                auto_broken += was_right and not now_right

    solved = path is not None and all(_inside(video, path[e], gt[e]) for e in range(n_events))
    return {
        "correct_at_start": correct_at_start, "solved": solved, "actions": actions,
        "auto_fixed": auto_fixed, "auto_broken": auto_broken,
    }


def auroc(margins_wrong: list[float], margins_right: list[float]) -> float:
    """Probability that a wrong event has a smaller margin than a correct one (ties count half)."""
    if not margins_wrong or not margins_right:
        return float("nan")
    wins = sum((w < r) + 0.5 * (w == r) for w in margins_wrong for r in margins_right)
    return wins / (len(margins_wrong) * len(margins_right))
