"""Deterministic candidate coverage selection for the AVS workflow.

This module owns temporal deduplication across nearby keyframes of the same video
and deterministic video-first pass ordering. It does not perform model inference,
temporal decoding, or learned reranking.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from hcmai.common.config import AvsConfig


@dataclass(frozen=True, slots=True)
class AvsCoverageCandidate:
    """Canonical representation of an AVS candidate evaluated for coverage selection."""

    frame_id: str
    video_id: str
    timestamp_ms: int
    retrieval_rank: int
    retrieval_score: float | None = None

    def __post_init__(self) -> None:
        if not self.frame_id.strip() or not self.video_id.strip():
            raise ValueError("AVS candidate identity must be non-empty")
        if self.timestamp_ms < 0:
            raise ValueError("AVS timestamp_ms must be non-negative")
        if self.retrieval_rank < 1:
            raise ValueError("AVS retrieval_rank must be one-based")


class AvsCoverageSelector:
    """Selects and orders candidates deterministically with same-video temporal dedup."""

    def __init__(self, config: AvsConfig) -> None:
        self.config = config

    def order(self, candidates: Sequence[AvsCoverageCandidate]) -> list[AvsCoverageCandidate]:
        ranked = sorted(candidates, key=lambda item: (item.retrieval_rank, item.frame_id))
        kept_by_video: dict[str, list[AvsCoverageCandidate]] = defaultdict(list)

        for candidate in ranked:
            if any(
                abs(candidate.timestamp_ms - kept.timestamp_ms)
                <= self.config.temporal_dedup_window_ms
                for kept in kept_by_video[candidate.video_id]
            ):
                continue
            kept_by_video[candidate.video_id].append(candidate)

        ordered: list[AvsCoverageCandidate] = []
        depth = 0
        while True:
            pass_items = [
                items[depth]
                for items in kept_by_video.values()
                if depth < len(items)
            ]
            if not pass_items:
                break
            ordered.extend(sorted(pass_items, key=lambda item: (item.retrieval_rank, item.frame_id)))
            depth += 1
        return ordered
