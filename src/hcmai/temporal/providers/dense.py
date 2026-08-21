"""Dense TRAKE event/frame evidence acquisition adapter."""

from __future__ import annotations

from typing import Any

from hcmai.common.schemas import TemporalAlignmentMode, TemporalQueryPlan
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class DenseOrderedEvidenceProvider:
    """Adapt visual video scoring to the ordered-evidence port."""

    def __init__(self, retrieval: RetrievalService, settings: Any) -> None:
        self.retrieval = retrieval
        self.settings = settings

    def acquire(
        self, plan: TemporalQueryPlan,
    ) -> tuple[VideoEventScores, ...]:
        if plan.alignment_mode is not TemporalAlignmentMode.ORDERED_PATH:
            raise ValueError("dense evidence requires an ordered-path plan")
        if plan.filters is not None:
            raise ValueError("dense ordered retrieval does not support filters")
        events = [unit.text for unit in plan.units]
        return tuple(self.retrieval.score_visual_videos(
            events,
            self.settings.top_k,
            self.settings.max_videos,
            self.settings.rrf_k,
            self.settings.chunk_size,
        ))
