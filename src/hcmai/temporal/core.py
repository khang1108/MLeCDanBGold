"""Shared progressive evidence acquisition, backfill, and scene localization."""

from __future__ import annotations

from dataclasses import dataclass, field

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameEvidence,
    QueryUnit,
    RetrievalTrace,
    SceneCandidate,
    SearchFilters,
    TaskType,
)
from hcmai.data.pipeline import DataService
from hcmai.retrieval.retriever.pipeline import RetrievalService

from .evidence import (
    ProgressiveEvidenceState,
    deduplicate_evidence,
    retrieval_to_evidence,
)
from .query import SnapshotDiffMode, SnapshotDiffResult, diff_snapshot
from .relations import parse_temporal_constraints
from .scoring import rank_scenes, score_scene
from .state import (
    ProgressiveSearchState,
    ProgressiveStateConflictError,
    ProgressiveStateStore,
)


@dataclass(frozen=True)
class ProgressiveLocalizationResult:
    """Expose committed state identity and ranked scenes to task-specific heads."""

    search_id: str
    version: int
    scenes: tuple[SceneCandidate, ...]
    diff: SnapshotDiffResult
    warnings: tuple[str, ...]
    diagnostics: dict[str, int | float]
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


class TemporalEvidenceCore:
    """Common core whose output boundary is ranked SceneCandidate objects."""

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: SearchConfig,
        store: ProgressiveStateStore | None = None,
    ) -> None:
        """Initialize the shared core and its bounded process-local state store."""

        self.data = data
        self.retrieval = retrieval
        self.config = config
        progressive = config.progressive
        self.store = store or ProgressiveStateStore(
            progressive.progressive_state_ttl_seconds,
            progressive.progressive_state_max_entries,
        )

    def localize(
        self,
        snapshot: str,
        *,
        search_id: str | None,
        filters: SearchFilters | None,
        task_type: TaskType = TaskType.KIS,
        session_fingerprint: str | None = None,
    ) -> ProgressiveLocalizationResult:
        """Process one cumulative snapshot and commit only successful changes."""

        identifier = search_id or self.store.new_id()
        with self.store.serialized(identifier):
            # A first request stays purely proposed until retrieval and scene
            # construction have both succeeded.
            if search_id is None:
                current = ProgressiveSearchState(
                    search_id=identifier,
                    task_type=task_type,
                    base_filters=(
                        filters.model_copy(deep=True) if filters is not None else None
                    ),
                    session_fingerprint=session_fingerprint,
                )
            else:
                current = self.store.get(identifier)
                self._validate_session(
                    current,
                    task_type=task_type,
                    filters=filters,
                    session_fingerprint=session_fingerprint,
                )
            diff = diff_snapshot(current.last_snapshot, snapshot)
            if diff.mode is SnapshotDiffMode.REPLACEMENT:
                raise ProgressiveStateConflictError(
                    "current snapshot is not a safe cumulative extension of "
                    "the committed snapshot"
                )
            if not diff.changed:
                return self._result(
                    current,
                    diff,
                    ("progressive_snapshot_no_change",),
                    RetrievalTrace(),
                )
            if (
                len(current.query_units)
                >= self.config.progressive.progressive_max_hints
            ):
                raise ProgressiveStateConflictError("progressive hint limit exceeded")

            # Work on an isolated clone so any retrieval/provider failure leaves
            # the last committed version untouched.
            proposed = current.clone()
            unit = QueryUnit(
                unit_id=f"h{len(proposed.query_units)}",
                text=diff.delta_text or "",
                order=len(proposed.query_units),
            )
            proposed.query_units.append(unit)
            proposed.last_snapshot = diff.normalized_current
            acquisition_warnings, acquisition_trace = self._acquire(
                proposed,
                unit,
                proposed.base_filters,
            )
            proposed.constraints = parse_temporal_constraints(
                proposed.query_units
            )
            proposed.ranked_scenes = self._assemble_and_score(proposed)
            if search_id is None:
                committed = self.store.create(proposed)
            else:
                committed = self.store.commit(
                    proposed,
                    expected_version=current.version,
                )
            warnings = tuple(dict.fromkeys(acquisition_warnings))
            return self._result(committed, diff, warnings, acquisition_trace)

    @staticmethod
    def _validate_session(
        state: ProgressiveSearchState,
        *,
        task_type: TaskType,
        filters: SearchFilters | None,
        session_fingerprint: str | None,
    ) -> None:
        """Reject continuations that change the progressive search universe."""

        if state.task_type is not task_type:
            raise ProgressiveStateConflictError(
                f"progressive search belongs to {state.task_type.value}, not "
                f"{task_type.value}"
            )
        if not _filters_equal(state.base_filters, filters):
            raise ProgressiveStateConflictError(
                "progressive search filters cannot change; start a new search"
            )
        if state.session_fingerprint != session_fingerprint:
            raise ProgressiveStateConflictError(
                "progressive search session context changed; start a new search"
            )

    def _acquire(
        self,
        state: ProgressiveSearchState,
        unit: QueryUnit,
        filters: SearchFilters | None,
    ) -> tuple[list[str], RetrievalTrace]:
        """Retrieve current-unit evidence globally, locally, and by backfill."""

        budget = self.config.progressive
        previous_videos = list(state.candidate_video_ids)

        # Global retrieval can rescue a video that was absent from earlier
        # rounds. Local retrieval reevaluates the current unit in prior videos.
        named_results = [
            ("global", self.retrieval.search(
                unit.text,
                top_k=budget.global_quota,
                filters=filters,
                query_type=state.task_type,
            ))
        ]
        if previous_videos:
            local_filters = _with_videos(filters, previous_videos)
            if local_filters.video_ids:
                named_results.append(
                    ("local", self.retrieval.search(
                        unit.text,
                        top_k=budget.local_quota,
                        filters=local_filters,
                        query_type=state.task_type,
                    ))
                )
        by_video: dict[str, list[FrameEvidence]] = {}
        trace = RetrievalTrace()
        for branch, result in named_results:
            trace = trace.merged(result.trace, prefix=branch)
            for candidate in result.candidates:
                item = retrieval_to_evidence(candidate, unit.unit_id, self.data)
                by_video.setdefault(item.frame.video_id, []).append(item)

        for video_id, items in by_video.items():
            retained = _retain_top_evidence(items, budget.top_m_evidence)
            state.evidence.mark_evaluated(unit.unit_id, video_id, retained)
        # Absence from a multi-video Top-K result proves nothing about an
        # individual video. Such pairs deliberately remain UNKNOWN.
        temporary_videos = list(dict.fromkeys((*previous_videos, *by_video)))
        rescued = [
            video_id
            for video_id in temporary_videos
            if video_id not in previous_videos
        ]
        warnings = [
            warning
            for _, result in named_results
            for warning in result.warnings
        ]
        backfill_targets = [
            *rescued,
            *(
                video_id
                for video_id in previous_videos
                if state.evidence.unknown_units(
                    [item.unit_id for item in state.query_units],
                    video_id,
                )
            ),
        ]
        backfill_warnings, backfill_trace = self._backfill(
                state,
                backfill_targets[: budget.backfill_max_videos],
                filters,
            )
        warnings.extend(backfill_warnings)
        trace = trace.merged(backfill_trace, prefix="backfill")
        candidate_scores = _candidate_video_scores(
            state.evidence,
            unit_ids=[item.unit_id for item in state.query_units],
            allowed_video_ids=set(temporary_videos),
            semantic_weight=budget.candidate_semantic_weight,
            match_weight=budget.candidate_match_weight,
            evaluation_weight=budget.candidate_evaluation_weight,
        )
        ranked_videos = sorted(
            temporary_videos,
            key=lambda key: (-candidate_scores.get(key, 0.0), key),
        )
        state.candidate_video_ids = ranked_videos[: budget.candidate_pool_size]
        state.evidence.retain_videos(set(state.candidate_video_ids))
        return warnings, trace

    def _backfill(
        self,
        state: ProgressiveSearchState,
        rescued_videos: list[str],
        filters: SearchFilters | None,
    ) -> tuple[list[str], RetrievalTrace]:
        """Evaluate bounded UNKNOWN older units for newly rescued videos."""

        budget = self.config.progressive
        warnings: list[str] = []
        trace = RetrievalTrace()
        unit_by_id = {unit.unit_id: unit for unit in state.query_units}
        ordered_ids = [unit.unit_id for unit in state.query_units]
        for video_id in rescued_videos:
            unknown = state.evidence.unknown_units(ordered_ids, video_id)
            for unit_id in unknown[: budget.backfill_max_units_per_video]:
                unit = unit_by_id[unit_id]
                result = self.retrieval.search(
                    unit.text,
                    top_k=budget.top_m_evidence,
                    filters=_with_videos(filters, [video_id]),
                    query_type=state.task_type,
                )
                warnings.extend(result.warnings)
                trace = trace.merged(
                    result.trace,
                    prefix=f"{video_id}.{unit_id}",
                )
                items = [
                    retrieval_to_evidence(candidate, unit_id, self.data)
                    for candidate in result.candidates
                    if self.data.get_frame(candidate.frame_id).video_id == video_id
                ]
                items = _retain_top_evidence(items, budget.top_m_evidence)
                state.evidence.mark_evaluated(unit_id, video_id, items)
        return warnings, trace

    def _assemble_and_score(
        self,
        state: ProgressiveSearchState,
    ) -> list[SceneCandidate]:
        """Build bounded per-video scenes and apply the shared scorer."""

        budget = self.config.progressive
        all_scenes: list[SceneCandidate] = []
        score_bounds = _unit_score_bounds(
            state.evidence,
            set(state.candidate_video_ids),
        )
        for video_id in state.candidate_video_ids:
            items = [
                item
                for (_, candidate_video), evidence in (
                    state.evidence.evidence.items()
                )
                if candidate_video == video_id
                for item in evidence
            ]
            scenes = _cluster_video_evidence(
                video_id,
                items,
                max_gap_ms=budget.scene_max_gap_ms,
                max_span_ms=budget.scene_max_span_ms,
            )
            scored = [
                score_scene(
                    scene,
                    state.query_units,
                    state.evidence,
                    state.constraints,
                    budget,
                    coherence_window_ms=budget.scene_coherence_ms,
                    unit_score_bounds=score_bounds,
                )
                for scene in scenes
            ]
            ranked = rank_scenes(scored)
            all_scenes.extend(ranked[: budget.scene_top_b_per_video])
        return rank_scenes(all_scenes)[: budget.scene_top_p_global]

    def _result(
        self,
        state: ProgressiveSearchState,
        diff: SnapshotDiffResult,
        warnings: tuple[str, ...],
        trace: RetrievalTrace,
    ) -> ProgressiveLocalizationResult:
        """Create the immutable task-boundary result from committed state."""

        return ProgressiveLocalizationResult(
            search_id=state.search_id,
            version=state.version,
            scenes=tuple(state.ranked_scenes),
            diff=diff,
            warnings=warnings,
            diagnostics=self.config.progressive.diagnostics(),
            trace=trace,
        )


def _with_videos(
    filters: SearchFilters | None,
    video_ids: list[str],
) -> SearchFilters:
    """Intersect a request filter with an explicit candidate-video subset."""

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


def _candidate_video_scores(
    state: ProgressiveEvidenceState,
    *,
    unit_ids: list[str],
    allowed_video_ids: set[str],
    semantic_weight: float,
    match_weight: float,
    evaluation_weight: float,
) -> dict[str, float]:
    """Rank videos by multi-unit quality and evaluation completeness."""

    bounds = _unit_score_bounds(state, allowed_video_ids)
    scores: dict[str, float] = {}
    for video_id in allowed_video_ids:
        evaluated = [
            unit_id
            for unit_id in unit_ids
            if state.is_evaluated(unit_id, video_id)
        ]
        matched_values: list[float] = []
        for unit_id in evaluated:
            items = state.get_evidence(unit_id, video_id)
            if items:
                matched_values.append(
                    _normalize_score(max(item.score for item in items), bounds[unit_id])
                )
        semantic = (
            sum(matched_values) / len(matched_values) if matched_values else 0.0
        )
        match_coverage = (
            len(matched_values) / len(evaluated) if evaluated else 0.0
        )
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


def _cluster_video_evidence(
    video_id: str,
    evidence: list[FrameEvidence],
    *,
    max_gap_ms: int,
    max_span_ms: int,
) -> list[SceneCandidate]:
    """Split evidence on both adjacent gap and total scene span."""

    unique: dict[str, FrameEvidence] = {}
    for item in evidence:
        prior = unique.get(item.frame.frame_id)
        if prior is None:
            unique[item.frame.frame_id] = item
            continue
        # The same canonical frame can support multiple query units. Merge its
        # provenance instead of discarding coverage when deduplicating.
        source_scores = {**prior.source_scores, **item.source_scores}
        source_ranks = {**prior.source_ranks, **item.source_ranks}
        unique[item.frame.frame_id] = prior.model_copy(
            update={
                "unit_scores": {**prior.unit_scores, **item.unit_scores},
                "source_scores": {
                    source: max(
                        prior.source_scores.get(source, score),
                        score,
                    )
                    for source, score in source_scores.items()
                },
                "source_ranks": {
                    source: min(
                        prior.source_ranks.get(source, rank),
                        rank,
                    )
                    for source, rank in source_ranks.items()
                },
                "score": max(prior.score, item.score),
                "provenance": tuple(
                    dict.fromkeys((*prior.provenance, *item.provenance))
                ),
            }
        )
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.frame.timestamp_ms,
            item.frame.frame_id,
        ),
    )
    if not ordered:
        return []
    clusters: list[list[FrameEvidence]] = [[ordered[0]]]
    for item in ordered[1:]:
        cluster_start = clusters[-1][0].frame.timestamp_ms
        prior_timestamp = clusters[-1][-1].frame.timestamp_ms
        within_gap = item.frame.timestamp_ms - prior_timestamp <= max_gap_ms
        within_span = item.frame.timestamp_ms - cluster_start <= max_span_ms
        if within_gap and within_span:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    scenes = []
    for cluster in clusters:
        start = cluster[0].frame.timestamp_ms
        end = cluster[-1].frame.timestamp_ms
        scenes.append(
            SceneCandidate(
                scene_id=f"{video_id}:{start}-{end}",
                video_id=video_id,
                start_ms=start,
                end_ms=end,
                evidence=tuple(cluster),
                reason_labels=("bounded_temporal_cluster",),
            )
        )
    return scenes


def _retain_top_evidence(
    items: list[FrameEvidence],
    limit: int,
) -> tuple[FrameEvidence, ...]:
    """Deduplicate canonical frames before applying the Top-M budget."""

    return deduplicate_evidence(tuple(items))[:limit]


def _unit_score_bounds(
    state: ProgressiveEvidenceState,
    allowed_video_ids: set[str],
) -> dict[str, tuple[float, float]]:
    """Return per-query-unit retrieval ranges for comparable semantics."""

    values: dict[str, list[float]] = {}
    for (unit_id, video_id), items in state.evidence.items():
        if video_id not in allowed_video_ids or not items:
            continue
        values.setdefault(unit_id, []).append(max(item.score for item in items))
    return {
        unit_id: (min(0.0, min(unit_values)), max(unit_values))
        for unit_id, unit_values in values.items()
    }


def _normalize_score(value: float, bounds: tuple[float, float]) -> float:
    """Normalize one value without pretending raw fusion scores are calibrated."""

    low, high = bounds
    if high <= low:
        return 1.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _filters_equal(
    left: SearchFilters | None,
    right: SearchFilters | None,
) -> bool:
    """Compare progressive search universes after canonical validation."""

    empty = SearchFilters().model_dump(mode="json")
    left_value = left.model_dump(mode="json") if left is not None else empty
    right_value = right.model_dump(mode="json") if right is not None else empty
    return left_value == right_value
