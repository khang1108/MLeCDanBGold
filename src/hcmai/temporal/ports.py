"""Typed boundaries for sparse scenes and dense ordered-path alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hcmai.common.schemas import (
    OrderedPathCandidate,
    QueryUnit,
    RetrievalTrace,
    SceneCandidate,
    SearchFilters,
    TemporalQueryPlan,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores

from .evidence import ProgressiveEvidenceState
from .state import ProgressiveSearchState


@dataclass(frozen=True, slots=True)
class ProgressiveAcquisition:
    """Proposed sparse evidence update without progressive-state commit."""

    evidence: ProgressiveEvidenceState
    candidate_video_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    trace: RetrievalTrace


class ProgressiveEvidenceProvider(Protocol):
    """Acquire bounded sparse evidence for one new progressive unit."""

    def acquire(
        self,
        state: ProgressiveSearchState,
        unit: QueryUnit,
        filters: SearchFilters | None,
    ) -> ProgressiveAcquisition: ...


class OrderedEvidenceProvider(Protocol):
    """Acquire dense event/frame scores for one ordered query plan."""

    def acquire(
        self, plan: TemporalQueryPlan,
    ) -> tuple[VideoEventScores, ...]: ...


class SceneAligner(Protocol):
    """Align sparse progressive evidence into bounded scenes."""

    def align(
        self,
        plan: TemporalQueryPlan,
        evidence: ProgressiveEvidenceState,
    ) -> tuple[SceneCandidate, ...]: ...


class OrderedPathAligner(Protocol):
    """Align dense scores into canonical monotonic frame paths."""

    def align(
        self,
        plan: TemporalQueryPlan,
        video_scores: tuple[VideoEventScores, ...],
        *,
        max_paths: int,
    ) -> tuple[OrderedPathCandidate, ...]: ...
