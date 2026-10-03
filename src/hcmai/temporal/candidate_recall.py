"""Candidate Recall@K evaluation for sparse candidate graph decoding.

Verifies Phase 8 of SOICT motion graph implementation plan:
Evaluates whether ground-truth event frames/windows are captured within top-K unary
candidate layers across K in {4, 8, 16, 32, 64}.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.transition_decoder import select_event_candidates


@dataclass(frozen=True, slots=True)
class CandidateRecallResult:
    """Summary metrics for one candidate budget K."""

    k: int
    recall: float
    total_events: int
    hit_events: int
    avg_edges_per_video: float
    avg_latency_ms: float

    def to_markdown_row(self) -> str:
        return f"| {self.k:4d} | {self.recall * 100:6.2f}% | {self.avg_edges_per_video:10.1f} | {self.avg_latency_ms:8.2f} ms |"


@dataclass(frozen=True, slots=True)
class EventAnnotation:
    """Ground truth annotation for one semantic event in a video."""

    video_id: str
    event_index: int
    target_frame_idxs: tuple[int, ...] = ()
    start_ms: int | None = None
    end_ms: int | None = None


def is_candidate_hit(
    candidate_frame_indices: np.ndarray,
    video: VideoEventScores,
    annotation: EventAnnotation,
) -> bool:
    """Check if any candidate frame matches ground truth by frame_idx or timestamp window."""
    if len(candidate_frame_indices) == 0:
        return False

    cand_frame_idxs = set(video.frame_idx[candidate_frame_indices].tolist())
    if annotation.target_frame_idxs and any(idx in cand_frame_idxs for idx in annotation.target_frame_idxs):
        return True

    if annotation.start_ms is not None and annotation.end_ms is not None:
        cand_ts = video.timestamps_ms[candidate_frame_indices]
        if np.any((cand_ts >= annotation.start_ms) & (cand_ts <= annotation.end_ms)):
            return True

    return False


def evaluate_candidate_recall(
    annotations: Sequence[EventAnnotation],
    video_scores: Mapping[str, VideoEventScores],
    k_values: Sequence[int] = (4, 8, 16, 32, 64),
    allowed: Mapping[str, np.ndarray] | None = None,
) -> list[CandidateRecallResult]:
    """Evaluate CandidateRecall@K across multiple candidate budgets K.

    Args:
        annotations: Sequence of EventAnnotation ground truths.
        video_scores: Mapping from video_id to VideoEventScores.
        k_values: Sequence of candidate budget integers.
        allowed: Optional per-video allowed mask mapping.

    Returns:
        List of CandidateRecallResult records for each K.
    """
    if not annotations:
        return []

    # Group annotations by video_id
    by_video: dict[str, list[EventAnnotation]] = {}
    for ann in annotations:
        by_video.setdefault(ann.video_id, []).append(ann)

    results: list[CandidateRecallResult] = []

    for k in k_values:
        total_hits = 0
        total_events = 0
        total_edges = 0
        latencies_ms: list[float] = []

        for video_id, video_anns in by_video.items():
            if video_id not in video_scores:
                continue
            video = video_scores[video_id]
            mask = allowed.get(video_id) if allowed else None

            start_t = perf_counter()
            cand_layers = select_event_candidates(
                video,
                candidate_k=k,
                allowed=mask,
            )
            latencies_ms.append((perf_counter() - start_t) * 1000.0)

            if len(cand_layers) == 0:
                total_events += len(video_anns)
                continue

            # Calculate graph edges: sum_{i=0}^{M-2} K_i * K_{i+1}
            edges = sum(
                len(cand_layers[i].frame_indices) * len(cand_layers[i + 1].frame_indices)
                for i in range(len(cand_layers) - 1)
            )
            total_edges += edges

            # Check hits for each event annotation
            for ann in video_anns:
                total_events += 1
                if 0 <= ann.event_index < len(cand_layers):
                    layer = cand_layers[ann.event_index]
                    if is_candidate_hit(layer.frame_indices, video, ann):
                        total_hits += 1

        recall = total_hits / total_events if total_events > 0 else 0.0
        n_vids = max(len(by_video), 1)
        avg_edges = total_edges / n_vids
        avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0

        results.append(
            CandidateRecallResult(
                k=k,
                recall=recall,
                total_events=total_events,
                hit_events=total_hits,
                avg_edges_per_video=avg_edges,
                avg_latency_ms=avg_lat,
            )
        )

    return results


def format_candidate_recall_table(results: Sequence[CandidateRecallResult]) -> str:
    """Format CandidateRecallResult list as a Markdown table."""
    lines = [
        "| K | Candidate Recall | Avg. edges/video | Latency |",
        "|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(r.to_markdown_row())
    return "\n".join(lines)


__all__ = [
    "CandidateRecallResult",
    "EventAnnotation",
    "evaluate_candidate_recall",
    "format_candidate_recall_table",
    "is_candidate_hit",
]
