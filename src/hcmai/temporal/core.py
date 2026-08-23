"""Shared facade for progressive scenes and ordered temporal paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    OrderedPathCandidate,
    QueryUnit,
    RetrievalTrace,
    SearchFilters,
    SceneCandidate,
    TaskType,
    TemporalAlignmentMode,
    TemporalConstraint,
    TemporalQueryPlan,
    TemporalRelation,
)
from hcmai.data.pipeline import DataService
from hcmai.pipelines.trake import TRAKESettings
from hcmai.retrieval.retriever.pipeline import RetrievalService

from .aligners import MonotonicOrderedPathAligner, ProgressiveSceneAligner
from .ports import (
    OrderedEvidenceProvider,
    OrderedPathAligner,
    ProgressiveEvidenceProvider,
    SceneAligner,
)
from .providers import DenseOrderedEvidenceProvider, ProgressiveEvidenceProvider
from .query import SnapshotDiffMode, SnapshotDiffResult, diff_snapshot
from .utils.relations import parse_temporal_constraints
from .state.state import (
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
    time_to_first_candidate_ms: float | None = None


@dataclass(frozen=True)
class OrderedAlignmentResult:
    """Expose one validated ordered plan and its canonical aligned paths."""

    plan: TemporalQueryPlan
    paths: tuple[OrderedPathCandidate, ...]
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, int | float | str] = field(default_factory=dict)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)


class TemporalEvidenceCore:
    """Compose typed evidence providers and aligners for every temporal task."""

    def __init__(
        self,
        data: DataService,
        retrieval: RetrievalService,
        config: SearchConfig,
        store: ProgressiveStateStore | None = None,
        *,
        progressive_provider: ProgressiveEvidenceProvider | None = None,
        scene_aligner: SceneAligner | None = None,
        ordered_provider: OrderedEvidenceProvider | None = None,
        ordered_aligner: OrderedPathAligner | None = None,
        trake_settings: TRAKESettings | None = None,
    ) -> None:
        """Initialize one facade shared by KIS and TRAKE task heads."""

        self.data = data
        self.retrieval = retrieval
        self.config = config
        progressive = config.progressive
        self.store = store or ProgressiveStateStore(
            progressive.progressive_state_ttl_seconds,
            progressive.progressive_state_max_entries,
        )
        settings = trake_settings or TRAKESettings()
        self.progressive_provider = progressive_provider or (
            ProgressiveEvidenceProvider(data, retrieval, progressive)
        )
        self.scene_aligner = scene_aligner or ProgressiveSceneAligner(progressive)
        self.ordered_provider = ordered_provider or DenseOrderedEvidenceProvider(
            retrieval, settings
        )
        self.ordered_aligner = ordered_aligner or MonotonicOrderedPathAligner(
            data, settings
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
                return self._progressive_result(
                    current,
                    diff,
                    ("progressive_snapshot_no_change",),
                    RetrievalTrace(),
                )

            if len(current.query_units) >= self.config.progressive.progressive_max_hints:
                raise ProgressiveStateConflictError("progressive hint limit exceeded")

            proposed = current.clone()
            unit = QueryUnit(
                unit_id=f"h{len(proposed.query_units)}",
                text=diff.delta_text or "",
                order=len(proposed.query_units),
            )

            proposed.query_units.append(unit)

            proposed.last_snapshot = diff.normalized_current
            proposed.constraints = parse_temporal_constraints(proposed.query_units)

            plan = TemporalQueryPlan(
                task_type=proposed.task_type,
                units=tuple(proposed.query_units),
                constraints=tuple(proposed.constraints),
                filters=proposed.base_filters,
                alignment_mode=TemporalAlignmentMode.PROGRESSIVE_SCENE,
            )
            acquisition = self.progressive_provider.acquire(
                proposed, unit, proposed.base_filters
            )

            proposed.evidence = acquisition.evidence
            proposed.candidate_video_ids = list(acquisition.candidate_video_ids)
            proposed.ranked_scenes = list(
                self.scene_aligner.align(plan, proposed.evidence)
            )

            committed = (
                self.store.create(proposed)
                if search_id is None
                else self.store.commit(proposed, expected_version=current.version)
            )

            return self._progressive_result(
                committed,
                diff,
                acquisition.warnings,
                acquisition.trace,
                time_to_first_candidate_ms=acquisition.time_to_first_candidate_ms,
            )

    def align_ordered(
        self,
        plan: TemporalQueryPlan,
        *,
        max_paths: int,
    ) -> OrderedAlignmentResult:
        """Acquire dense evidence and align one explicit ordered-path plan."""

        if max_paths <= 0:
            raise ValueError("max_paths must be greater than zero")
        if plan.alignment_mode is not TemporalAlignmentMode.ORDERED_PATH:
            raise ValueError("ordered alignment requires an ordered-path plan")
        scores = self.ordered_provider.acquire(plan)
        paths = self.ordered_aligner.align(plan, scores, max_paths=max_paths)
        return OrderedAlignmentResult(
            plan=plan,
            paths=paths,
            diagnostics={
                "alignment_mode": plan.alignment_mode.value,
                "query_unit_count": len(plan.units),
                "candidate_video_count": len(scores),
                "path_count": len(paths),
            },
        )

    @staticmethod
    def ordered_plan(events: list[str]) -> TemporalQueryPlan:
        """Build the validated stateless TRAKE plan from explicit events."""

        units = tuple(
            QueryUnit(unit_id=f"e{index}", text=text, order=index)
            for index, text in enumerate(events)
        )
        constraints = tuple(
            TemporalConstraint(
                relation=TemporalRelation.BEFORE,
                subject_unit_id=units[index - 1].unit_id,
                object_unit_id=unit.unit_id,
                reason="ordered_task_contract",
            )
            for index, unit in enumerate(units)
            if index
        )
        return TemporalQueryPlan(
            task_type=TaskType.TRAKE,
            units=units,
            constraints=constraints,
            alignment_mode=TemporalAlignmentMode.ORDERED_PATH,
        )

    @staticmethod
    def _validate_session(
        state: ProgressiveSearchState,
        *,
        task_type: TaskType,
        filters: SearchFilters | None,
        session_fingerprint: str | None,
    ) -> None:
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

    def _progressive_result(
        self,
        state: ProgressiveSearchState,
        diff: SnapshotDiffResult,
        warnings: tuple[str, ...],
        trace: RetrievalTrace,
        time_to_first_candidate_ms: float | None = None,
    ) -> ProgressiveLocalizationResult:
        return ProgressiveLocalizationResult(
            search_id=state.search_id,
            version=state.version,
            scenes=tuple(state.ranked_scenes),
            diff=diff,
            warnings=warnings,
            diagnostics=self.config.progressive.diagnostics(),
            trace=trace,
            time_to_first_candidate_ms=time_to_first_candidate_ms,
        )


def _filters_equal(
    left: SearchFilters | None, right: SearchFilters | None,
) -> bool:
    empty = SearchFilters().model_dump(mode="json")
    left_value = left.model_dump(mode="json") if left is not None else empty
    right_value = right.model_dump(mode="json") if right is not None else empty
    return left_value == right_value
