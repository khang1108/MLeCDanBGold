"""Evaluation metrics for video retrieval, event grounding, and full path accuracy.

Verifies Phase 13 of SOICT motion graph implementation plan:
- Video Retrieval: R@1, R@5, R@10, MRR
- Event Grounding: EventHit@delta, AllHit@delta, Mean Hit Rate
- Diagnostics: Reverse Accuracy, Shuffled Margin
- Latency and evaluated edges
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticQuery
from hcmai.temporal.dp import DPPath


@dataclass(frozen=True, slots=True)
class PathGroundingResult:
    """Grounding evaluation for one decoded path against query event annotations."""

    query_id: str
    video_id: str
    all_hit: bool
    event_hits: tuple[bool, ...]
    hit_rate: float


@dataclass(frozen=True, slots=True)
class RetrievalMetricsResult:
    """Ranking accuracy for one target video among candidate video paths."""

    r1: float
    r5: float
    r10: float
    mrr: float
    rank: int | None


@dataclass(frozen=True, slots=True)
class MethodEvaluationSummary:
    """Aggregate benchmark metrics for one evaluation condition."""

    method_name: str
    num_queries: int
    r1: float
    r5: float
    r10: float
    mrr: float
    event_hit: float
    all_hit: float
    reverse_acc: float = 0.0
    avg_edges_per_video: float = 0.0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "num_queries": self.num_queries,
            "r1": self.r1,
            "r5": self.r5,
            "r10": self.r10,
            "mrr": self.mrr,
            "event_hit": self.event_hit,
            "all_hit": self.all_hit,
            "reverse_acc": self.reverse_acc,
            "avg_edges_per_video": self.avg_edges_per_video,
            "avg_latency_ms": self.avg_latency_ms,
        }

    def to_table_row(self) -> str:
        return (
            f"| {self.method_name:<20} "
            f"| {self.r1 * 100:5.1f}% "
            f"| {self.r5 * 100:5.1f}% "
            f"| {self.mrr:5.3f} "
            f"| {self.event_hit * 100:5.1f}% "
            f"| {self.all_hit * 100:5.1f}% "
            f"| {self.reverse_acc * 100:5.1f}% |"
        )


def evaluate_path_grounding(
    path: DPPath,
    query: DiagnosticQuery,
    video: VideoEventScores,
    delta_ms: int = 1000,
) -> PathGroundingResult:
    """Evaluate whether each aligned event frame falls within ground-truth coordinates."""
    n_events = len(query.events)
    if len(path.frame_idx) != n_events:
        return PathGroundingResult(
            query_id=query.query_id,
            video_id=query.video_id,
            all_hit=False,
            event_hits=tuple(False for _ in range(n_events)),
            hit_rate=0.0,
        )

    # Build mapping from frame_idx to timestamp_ms
    idx_to_time = {int(idx): int(ts) for idx, ts in zip(video.frame_idx, video.timestamps_ms)}

    event_hits: list[bool] = []
    for ev_idx, ev in enumerate(query.events):
        aligned_idx = int(path.frame_idx[ev_idx])
        hit = False

        # 1. Exact frame coordinate match
        if ev.target_frame_idxs and aligned_idx in ev.target_frame_idxs:
            hit = True
        # 2. Temporal window match with tolerance delta_ms
        elif ev.start_ms is not None and ev.end_ms is not None:
            aligned_time = idx_to_time.get(aligned_idx)
            if aligned_time is not None:
                if (ev.start_ms - delta_ms) <= aligned_time <= (ev.end_ms + delta_ms):
                    hit = True

        event_hits.append(hit)

    hits_tuple = tuple(event_hits)
    all_hit = all(hits_tuple)
    hit_rate = sum(hits_tuple) / len(hits_tuple) if hits_tuple else 0.0

    return PathGroundingResult(
        query_id=query.query_id,
        video_id=query.video_id,
        all_hit=all_hit,
        event_hits=hits_tuple,
        hit_rate=hit_rate,
    )


def evaluate_retrieval_ranking(
    ranked_paths: Sequence[DPPath],
    target_video_id: str,
) -> RetrievalMetricsResult:
    """Compute R@1, R@5, R@10, and MRR for a target video."""
    seen_videos: list[str] = []
    for p in ranked_paths:
        if p.video_id not in seen_videos:
            seen_videos.append(p.video_id)

    rank = None
    if target_video_id in seen_videos:
        rank = seen_videos.index(target_video_id) + 1  # 1-indexed

    r1 = 1.0 if rank == 1 else 0.0
    r5 = 1.0 if rank is not None and rank <= 5 else 0.0
    r10 = 1.0 if rank is not None and rank <= 10 else 0.0
    mrr = (1.0 / rank) if rank is not None else 0.0

    return RetrievalMetricsResult(r1=r1, r5=r5, r10=r10, mrr=mrr, rank=rank)


__all__ = [
    "MethodEvaluationSummary",
    "PathGroundingResult",
    "RetrievalMetricsResult",
    "evaluate_path_grounding",
    "evaluate_retrieval_ranking",
]
