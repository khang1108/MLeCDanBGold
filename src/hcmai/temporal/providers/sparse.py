"""Sparse progressive evidence acquisition for multi-round video retrieval.

Module Overview:
----------------
This module implements the sparse evidence acquisition pipeline used in progressive
search tasks (KIS and VQA). As hints/clues are revealed sequentially over time
(Hint 1 -> Hint 2 -> ... -> Hint N), this provider incrementally accumulates frame-level
evidence across candidate videos.

Key Algorithms & Strategies:
----------------------------
1. Dual-Branch Retrieval (Global + Local):
    - Global Search: Explores the entire video corpus with the newest hint to discover
        potential new candidate videos.
    - Local Search: Focuses retrieval of the newest hint specifically on video candidates
        identified in earlier rounds to verify continuity and deepen evidence.

2. Rescued Video Backfill:
    - When a video suddenly emerges with high relevance at a later hint (a "rescued" video),
        it lacks evaluation records for earlier hints (UNKNOWN state).
    - The `_backfill` routine automatically queries missing past hints targeted only at
        that video, preventing "missing evidence" from being penalized as "negative evidence".

3. Multi-Criteria Candidate Video Scoring:
    - Evaluates video quality based on normalized semantic match scores, match coverage
        across evaluated hints, and overall evaluation completeness.

4. Bounded Top-M Evidence Retention:
    - Retains a bounded set of top-M frames per hint per video (avoiding global best-1 traps
        and unbounded memory growth) for subsequent temporal scene alignment.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hcmai.common.config import ProgressiveSearchConfig
from hcmai.common.schemas import (
    FrameEvidence,
    QueryUnit,
    RetrievalResult,
    RetrievalTrace,
    SearchFilters,
)
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.temporal.evidence import (
    ProgressiveEvidenceState,
    deduplicate_evidence,
    retrieval_to_evidence,
)
from hcmai.temporal.ports import ProgressiveAcquisition
from hcmai.temporal.scoring import normalize_score, unit_score_bounds
from hcmai.temporal.state import ProgressiveSearchState


class ProgressiveEvidenceProvider:
    """Acquires frame evidence across progressive hints without mutating state.

    This provider coordinates multi-branch retrieval, converts raw retrieval candidates
    into canonical `FrameEvidence` objects, triggers backfills for unevaluated historical
    hints, and scores candidate videos for pruning.

    Attributes:
        data: Authoritative canonical data service for resolving frame metadata.
        retrieval: Multimodal retrieval service for visual/caption/OCR/ASR searching.
        config: Configuration parameters governing search quotas, weights, and limits.
    """

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: ProgressiveSearchConfig,
    ) -> None:
        """Initialize the progressive evidence provider.

        Args:
            data: Canonical data access service for frame metadata and enrichment text.
            retrieval: Retrieval pipeline service for multimodal vector/sparse search.
            config: Configuration settings for progressive acquisition quotas and weights.
        """
        self.data = data
        self.retrieval = retrieval
        self.config = config

    def acquire(
        self,
        state: ProgressiveSearchState,
        unit: QueryUnit,
        filters: SearchFilters | None,
    ) -> ProgressiveAcquisition:
        """Acquire evidence for a new query unit and produce updated candidate videos.

        Args:
            state: Current progressive search state holding historical units and evidence.
            unit: The latest query unit / hint to be evaluated.
            filters: Optional user-supplied filters (time ranges, video ID whitelist).

        Returns:
            A `ProgressiveAcquisition` object containing updated evidence, candidate video IDs,
            merged observability traces, and latency metrics.
        """
        # Step 1: Deepcopy evidence state to ensure isolation until caller commits
        evidence = deepcopy(state.evidence)
        previous_videos = list(state.candidate_video_ids)

        def _to_result(raw: Any) -> RetrievalResult:
            """Normalize raw retriever return value into a RetrievalResult."""
            if isinstance(raw, RetrievalResult):
                return raw
            return RetrievalResult(candidates=list(raw))

        # Step 2: Global search across entire corpus for the new hint
        raw_global = self.retrieval.search(
            unit.text,
            top_k=self.config.global_quota,
            filters=filters,
            query_type=state.task_type,
        )
        named_results = [("global", _to_result(raw_global))]

        # Step 3: Local search targeting candidate videos from previous hints
        if previous_videos:
            local_filters = with_videos(filters, previous_videos)
            if local_filters.video_ids:
                raw_local = self.retrieval.search(
                    unit.text,
                    top_k=self.config.local_quota,
                    filters=local_filters,
                    query_type=state.task_type,
                )
                named_results.append(("local", _to_result(raw_local)))

        # Step 4: Group retrieved candidates by video and convert to FrameEvidence
        by_video: dict[str, list[FrameEvidence]] = {}
        trace = RetrievalTrace()

        for branch, result in named_results:
            trace = trace.merged(result.trace, prefix=branch)
            
            for candidate in result.candidates:
                item = retrieval_to_evidence(candidate, unit.unit_id, self.data)
                by_video.setdefault(item.frame.video_id, []).append(item)

        # Mark evaluated and retain top-M evidence per video for this unit
        for video_id, items in by_video.items():
            evidence.mark_evaluated(
                unit.unit_id,
                video_id,
                retain_top_evidence(items, self.config.top_m_evidence),
            )

        # Step 5: Identify rescued videos and videos needing backfill
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

        # Step 6: Execute backfill retrieval for missing historical hint units
        backfill_warnings, backfill_trace = self._backfill(
            evidence,
            state,
            backfill_targets[: self.config.backfill_max_videos],
            filters,
        )
        warnings.extend(backfill_warnings)
        trace = trace.merged(backfill_trace, prefix="backfill")

        # Step 7: Score candidate videos based on semantic quality and coverage
        scores = candidate_video_scores(
            evidence,
            unit_ids=[item.unit_id for item in state.query_units],
            allowed_video_ids=set(temporary_videos),
            semantic_weight=self.config.candidate_semantic_weight,
            match_weight=self.config.candidate_match_weight,
            evaluation_weight=self.config.candidate_evaluation_weight,
        )

        # Step 8: Rank and prune candidate video pool
        ranked = sorted(
            temporary_videos,
            key=lambda key: (-scores.get(key, 0.0), key),
        )[: self.config.candidate_pool_size]
        evidence.retain_videos(set(ranked))

        # Extract time to first candidate latency metric if available
        time_to_first_candidate_ms = next(
            (
                result.time_to_first_candidate_ms
                for _, result in named_results
                if getattr(result, "time_to_first_candidate_ms", None) is not None
            ),
            None,
        )

        return ProgressiveAcquisition(
            evidence=evidence,
            candidate_video_ids=tuple(ranked),
            warnings=tuple(dict.fromkeys(warnings)),
            trace=trace,
            time_to_first_candidate_ms=time_to_first_candidate_ms,
        )

    def _backfill(
        self,
        evidence: ProgressiveEvidenceState,
        state: ProgressiveSearchState,
        videos: list[str],
        filters: SearchFilters | None,
    ) -> tuple[list[str], RetrievalTrace]:
        """Perform targeted backfill searches for videos with un-evaluated past hints.

        When a video enters the candidate set at hint K (K > 1), hints 1..(K-1) were
        never evaluated for that video (state is UNKNOWN). This method searches specifically
        for those missing hint units restricted to that video, ensuring a fair evaluation.

        Args:
            evidence: Mutable progressive evidence state being updated.
            state: Current search state containing all historical query units.
            videos: List of target video IDs requiring backfill.
            filters: Optional base search filters to combine with video ID restriction.

        Returns:
            A tuple of (warnings_list, merged_retrieval_trace).
        """
        warnings: list[str] = []
        trace = RetrievalTrace()
        unit_by_id = {unit.unit_id: unit for unit in state.query_units}
        ordered_ids = [unit.unit_id for unit in state.query_units]

        for video_id in videos:
            unknown = evidence.unknown_units(ordered_ids, video_id)
            for unit_id in unknown[: self.config.backfill_max_units_per_video]:
                raw_result = self.retrieval.search(
                    unit_by_id[unit_id].text,
                    top_k=self.config.top_m_evidence,
                    filters=with_videos(filters, [video_id]),
                    query_type=state.task_type,
                )
                result = (
                    raw_result
                    if isinstance(raw_result, RetrievalResult)
                    else RetrievalResult(candidates=list(raw_result))
                )
                warnings.extend(result.warnings)
                trace = trace.merged(result.trace, prefix=f"{video_id}.{unit_id}")

                # Convert candidates to FrameEvidence and filter strictly for this video
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
    filters: SearchFilters | None,
    video_ids: list[str],
) -> SearchFilters:
    """Intersect incoming search filters with an explicit whitelist of candidate video IDs.

    Args:
        filters: Existing search filters (e.g. from API request), or None.
        video_ids: Whitelist of video IDs to restrict retrieval to.

    Returns:
        A new SearchFilters instance containing the intersection of allowed video IDs
        and preserving all other filter constraints (time ranges, score thresholds).
    """
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
    """Rank candidate videos using a balanced multi-criteria scoring formulation.

    Scoring Formula:
    ----------------
    For each video V:
        - semantic: Average min-max normalized similarity score across all matched units.
        - match_coverage: Fraction of evaluated units that yielded valid evidence matches.
        - evaluation_coverage: Fraction of total units evaluated for this video.

        Score(V) = (w_sem * semantic + w_match * match_cov * eval_cov + w_eval * eval_cov)
                ------------------------------------------------------------------------
                                    (w_sem + w_match + w_eval)

    This balances raw retrieval similarity against consistency across multiple hints,
    while rewarding videos that have completed full backfill evaluation.

    Args:
        state: The current progressive evidence state.
        unit_ids: Ordered list of all query unit IDs up to the current hint.
        allowed_video_ids: Set of video IDs to score.
        semantic_weight: Weight assigned to raw semantic similarity quality.
        match_weight: Weight assigned to cross-hint match coverage consistency.
        evaluation_weight: Weight assigned to evaluation completeness.

    Returns:
        Dictionary mapping each video_id to its composite score in [0.0, 1.0].
    """
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
        total_weight = semantic_weight + match_weight + evaluation_weight
        scores[video_id] = weighted / total_weight if total_weight > 0 else 0.0

    return scores


def retain_top_evidence(
    items: list[FrameEvidence],
    limit: int,
) -> tuple[FrameEvidence, ...]:
    """Deduplicate and retain the top-M highest scoring frame evidence objects.

    Args:
        items: List of frame evidence objects to prune.
        limit: Maximum number of evidence frames to retain (top-M).

    Returns:
        A tuple of deduplicated and pruned FrameEvidence items.
    """
    return deduplicate_evidence(tuple(items))[:limit]
