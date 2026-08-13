"""Sparse global/local/backfill evidence acquisition for progressive search."""

from __future__ import annotations

from copy import deepcopy

from hcmai.common.config import ProgressiveSearchConfig
from hcmai.common.schemas import FrameEvidence, QueryUnit, RetrievalTrace, SearchFilters
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService

from ..evidence import (
    ProgressiveEvidenceState,
    deduplicate_evidence,
    retrieval_to_evidence,
)
from ..ports import ProgressiveAcquisition
from ..scoring import normalize_score, unit_score_bounds
from ..state import ProgressiveSearchState


class SparseProgressiveEvidenceProvider:
    """Acquire sparse evidence without owning progressive-state commits."""

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: ProgressiveSearchConfig,
    ) -> None:
        self.data = data
        self.retrieval = retrieval
        self.config = config

    def acquire(
        self,
        state: ProgressiveSearchState,
        unit: QueryUnit,
        filters: SearchFilters | None,
    ) -> ProgressiveAcquisition:
        evidence = deepcopy(state.evidence)
        previous_videos = list(state.candidate_video_ids)
        named_results = [("global", self.retrieval.search(
            unit.text,
            top_k=self.config.global_quota,
            filters=filters,
            query_type=state.task_type,
        ))]
        if previous_videos:
            local_filters = with_videos(filters, previous_videos)
            if local_filters.video_ids:
                named_results.append(("local", self.retrieval.search(
                    unit.text,
                    top_k=self.config.local_quota,
                    filters=local_filters,
                    query_type=state.task_type,
                )))

        by_video: dict[str, list[FrameEvidence]] = {}
        trace = RetrievalTrace()
        for branch, result in named_results:
            trace = trace.merged(result.trace, prefix=branch)
            for candidate in result.candidates:
                item = retrieval_to_evidence(candidate, unit.unit_id, self.data)
                by_video.setdefault(item.frame.video_id, []).append(item)
        for video_id, items in by_video.items():
            evidence.mark_evaluated(
                unit.unit_id,
                video_id,
                retain_top_evidence(items, self.config.top_m_evidence),
            )

        temporary_videos = list(dict.fromkeys((*previous_videos, *by_video)))
        rescued = [
            video_id for video_id in temporary_videos
            if video_id not in previous_videos
        ]
        warnings = [
            warning for _, result in named_results for warning in result.warnings
        ]
        backfill_targets = [
            *rescued,
            *(
                video_id
                for video_id in previous_videos
                if evidence.unknown_units(
                    [item.unit_id for item in state.query_units], video_id
                )
            ),
        ]
        backfill_warnings, backfill_trace = self._backfill(
            evidence,
            state,
            backfill_targets[: self.config.backfill_max_videos],
            filters,
        )
        warnings.extend(backfill_warnings)
        trace = trace.merged(backfill_trace, prefix="backfill")
        scores = candidate_video_scores(
            evidence,
            unit_ids=[item.unit_id for item in state.query_units],
            allowed_video_ids=set(temporary_videos),
            semantic_weight=self.config.candidate_semantic_weight,
            match_weight=self.config.candidate_match_weight,
            evaluation_weight=self.config.candidate_evaluation_weight,
        )
        ranked = sorted(
            temporary_videos,
            key=lambda key: (-scores.get(key, 0.0), key),
        )[: self.config.candidate_pool_size]
        evidence.retain_videos(set(ranked))
        return ProgressiveAcquisition(
            evidence=evidence,
            candidate_video_ids=tuple(ranked),
            warnings=tuple(dict.fromkeys(warnings)),
            trace=trace,
        )

    def _backfill(
        self,
        evidence: ProgressiveEvidenceState,
        state: ProgressiveSearchState,
        videos: list[str],
        filters: SearchFilters | None,
    ) -> tuple[list[str], RetrievalTrace]:
        warnings: list[str] = []
        trace = RetrievalTrace()
        unit_by_id = {unit.unit_id: unit for unit in state.query_units}
        ordered_ids = [unit.unit_id for unit in state.query_units]
        for video_id in videos:
            unknown = evidence.unknown_units(ordered_ids, video_id)
            for unit_id in unknown[: self.config.backfill_max_units_per_video]:
                result = self.retrieval.search(
                    unit_by_id[unit_id].text,
                    top_k=self.config.top_m_evidence,
                    filters=with_videos(filters, [video_id]),
                    query_type=state.task_type,
                )
                warnings.extend(result.warnings)
                trace = trace.merged(result.trace, prefix=f"{video_id}.{unit_id}")
                items = [
                    retrieval_to_evidence(candidate, unit_id, self.data)
                    for candidate in result.candidates
                    if self.data.get_frame(candidate.frame_id).video_id == video_id
                ]
                evidence.mark_evaluated(
                    unit_id,
                    video_id,
                    retain_top_evidence(items, self.config.top_m_evidence),
                )
        return warnings, trace


def with_videos(
    filters: SearchFilters | None, video_ids: list[str],
) -> SearchFilters:
    """Intersect request filters with one explicit candidate-video subset."""

    allowed = list(video_ids)
    if filters is not None and filters.video_ids:
        requested = set(filters.video_ids)
        allowed = [video_id for video_id in allowed if video_id in requested]
    return SearchFilters(
        video_ids=allowed,
        start_time_ms=filters.start_time_ms if filters else None,
        end_time_ms=filters.end_time_ms if filters else None,
        min_score=filters.min_score if filters else None,
    )


def candidate_video_scores(
    state: ProgressiveEvidenceState,
    *,
    unit_ids: list[str],
    allowed_video_ids: set[str],
    semantic_weight: float,
    match_weight: float,
    evaluation_weight: float,
) -> dict[str, float]:
    """Rank videos by multi-unit quality and evaluation completeness."""

    bounds = unit_score_bounds(state, allowed_video_ids)
    scores: dict[str, float] = {}
    for video_id in allowed_video_ids:
        evaluated = [
            unit_id for unit_id in unit_ids
            if state.is_evaluated(unit_id, video_id)
        ]
        matched_values = [
            normalize_score(
                max(item.score for item in state.get_evidence(unit_id, video_id)),
                bounds[unit_id],
            )
            for unit_id in evaluated
            if state.get_evidence(unit_id, video_id)
        ]
        semantic = sum(matched_values) / len(matched_values) if matched_values else 0.0
        match_coverage = len(matched_values) / len(evaluated) if evaluated else 0.0
        evaluation_coverage = len(evaluated) / len(unit_ids) if unit_ids else 0.0
        weighted = (
            semantic_weight * semantic
            + match_weight * match_coverage * evaluation_coverage
            + evaluation_weight * evaluation_coverage
        )
        scores[video_id] = weighted / (
            semantic_weight + match_weight + evaluation_weight
        )
    return scores


def retain_top_evidence(
    items: list[FrameEvidence], limit: int,
) -> tuple[FrameEvidence, ...]:
    return deduplicate_evidence(tuple(items))[:limit]
