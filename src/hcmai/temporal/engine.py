"""Run one progressive snapshot end to end: retrieve, backfill, assemble, score."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import UUID

from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.search import SearchFilters
from hcmai.common.schemas.telemetry import RetrievalTrace
from hcmai.temporal.alignment.coverage import CoverageWindowAligner
from hcmai.temporal.backfill import LazyBackfillRequest, backfill_rescued_videos
from hcmai.temporal.evidence import EvidenceStore
from hcmai.temporal.lifecycle import prepare_snapshot
from hcmai.temporal.models import EvidencePoint, EvidenceStatus, SceneCandidate
from hcmai.temporal.plan import TemporalQueryPlan
from hcmai.temporal.query import RuleTemporalRelationParser
from hcmai.temporal.retrieval import (
    SparseEvidenceProvider,
    SparseEvidenceRequest,
    gather_evidence,
)
from hcmai.temporal.scoring import SceneScorer
from hcmai.temporal.state import ProgressiveSearchState, ProgressiveStateStore


@dataclass(frozen=True, slots=True)
class TemporalSearchResult:
    """Ranked scenes plus the state the caller should commit."""

    search_id: UUID
    scenes: tuple[SceneCandidate, ...]
    state: ProgressiveSearchState
    commit_required: bool
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalEvidenceEngine:
    """Locate scenes for KIS/VQA; never materializes frames and never answers a question."""

    provider: SparseEvidenceProvider
    states: ProgressiveStateStore
    aligner: CoverageWindowAligner = CoverageWindowAligner()
    scorer: SceneScorer = SceneScorer()
    parser: RuleTemporalRelationParser = RuleTemporalRelationParser()
    top_m: int = 10
    top_k: int = 100
    global_quota: int = 100
    local_quota: int = 100
    max_total: int = 20

    def __post_init__(self) -> None:
        if self.max_total < 1:
            raise ValueError("max_total must be at least 1")

    def search(
        self,
        snapshot: str,
        *,
        task_type: TaskType = TaskType.KIS,
        search_id: UUID | None = None,
        filters: SearchFilters | None = None,
        allow_missing_state_fallback: bool = False,
    ) -> TemporalSearchResult:
        """Fold a cumulative snapshot into the session and re-rank its scenes."""
        prepared = prepare_snapshot(
            self.states,
            task_type,
            snapshot,
            search_id=search_id,
            allow_missing_state_fallback=allow_missing_state_fallback,
        )
        warnings = list(prepared.warnings)
        refiltered = (filters or SearchFilters()) != (
            prepared.state.filters or SearchFilters()
        )
        if prepared.delta is None and not refiltered:
            return TemporalSearchResult(
                search_id=prepared.state.search_id,
                scenes=prepared.state.scene_candidates[: self.max_total],
                state=prepared.state,
                commit_required=False,
                warnings=tuple(warnings),
            )

        units = prepared.state.query_units
        store = EvidenceStore(top_m=self.top_m)
        if refiltered:
            replayed = list(enumerate(units))
        else:
            for (unit_id, video_id), evidence in prepared.state.evidence_by_unit_video.items():
                store.record(unit_id, video_id, evidence.points)
            replayed = [(len(units) - 1, units[-1])]

        trace = RetrievalTrace()
        for index, unit in replayed:
            gathered = gather_evidence(
                self.provider,
                store,
                SparseEvidenceRequest(
                    unit_id=unit.unit_id,
                    query=unit.text,
                    top_k=self.top_k,
                    filters=filters,
                    task_type=task_type,
                ),
                global_quota=self.global_quota,
                local_quota=self.local_quota,
            )
            warnings.extend(gathered.warnings)
            trace = trace.merged(gathered.trace, prefix=unit.unit_id)
            backfilled = backfill_rescued_videos(
                self.provider,
                store,
                LazyBackfillRequest(
                    old_units=units[:index],
                    rescued_video_ids=gathered.rescued_video_ids,
                    top_k=self.top_k,
                    filters=filters,
                    task_type=task_type,
                ),
            )
            warnings.extend(backfilled.warnings)

        parsed = self.parser.parse(units)
        warnings.extend(parsed.warnings)
        plan = TemporalQueryPlan(
            units=units,
            constraints=parsed.constraints,
            filters=filters,
        )

        points: list[EvidencePoint] = []
        evidence_by_unit_video = {}
        evaluated_units_by_video: dict[str, set[str]] = {}
        video_ids = store.video_ids()
        for unit in plan.units:
            for video_id in video_ids:
                evidence = store.get(unit.unit_id, video_id)
                if evidence.status is EvidenceStatus.UNKNOWN:
                    continue
                evidence_by_unit_video[(unit.unit_id, video_id)] = evidence
                evaluated_units_by_video.setdefault(video_id, set()).add(unit.unit_id)
                points.extend(evidence.points)

        scenes = self.scorer.rank(plan, self.aligner.align(plan, points).candidates)
        state = replace(
            prepared.state,
            filters=filters,
            constraints=parsed.constraints,
            evaluated_units_by_video={
                video_id: frozenset(unit_ids)
                for video_id, unit_ids in evaluated_units_by_video.items()
            },
            evidence_by_unit_video=evidence_by_unit_video,
            scene_candidates=scenes,
        )
        return TemporalSearchResult(
            search_id=state.search_id,
            scenes=scenes[: self.max_total],
            state=state,
            commit_required=True,
            trace=trace,
            warnings=tuple(warnings),
        )
