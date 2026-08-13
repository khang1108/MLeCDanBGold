"""Current frame-search behavior exposed as an executable task pipeline."""

from __future__ import annotations

from time import perf_counter

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    RetrievalTrace,
    RetrievalCandidate,
    RetrievalResult,
    SearchRequest,
    SearchResponse,
    TaskRequest,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.pipelines.kis.ranking import KISRankingConfig, shape_kis_candidates
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.orchestration.ranking import elapsed_ms, rank_candidates, request_id
from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer, log_stage
from hcmai.retrieval.reranking.pipeline import RerankingService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.temporal import TemporalEvidenceCore

logger = get_logger(__name__)


class KISPipeline:
    """Adapt the existing KIS ranking and materialization behavior."""

    def __init__(
        self,
        task_type: TaskType,
        data: DataService | None,
        retrieval: RetrievalService | None,
        reranking: RerankingService | None,
        config: SearchConfig,
        temporal_core: TemporalEvidenceCore | None = None,
    ) -> None:
        """Initialize the KIS head and its shared localization dependency."""

        if task_type not in {TaskType.KIS, TaskType.VKIS}:
            raise ValueError(f"KISPipeline cannot handle {task_type.value!r}")
        self._task_type = task_type
        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config
        self.temporal_core = temporal_core
        self.materializer = SearchMaterializer(data) if data is not None else None

    @property
    def task_type(self) -> TaskType:
        """Return the concrete KIS-family task handled by this pipeline."""

        return self._task_type

    def execute(self, request: TaskRequest) -> SearchResponse:
        """Localize scenes, select representative frames, and materialize them."""

        if not isinstance(request, SearchRequest):
            raise TaskPipelineRequestError(
                f"pipeline for {self.task_type.value!r} requires a search request"
            )
        if request.query_type is not self.task_type:
            raise TaskPipelineRequestError(
                f"pipeline for {self.task_type.value!r} cannot execute "
                f"request for {request.query_type.value!r}"
            )
        if self.data is None or self.materializer is None:
            raise TaskPipelineDependencyError("Frame store not loaded")
        if self.retrieval is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        started = perf_counter()
        parse_trace = StageTimer(PipelineStage.PARSE.value).finish(
            input_count=1,
            output_count=1,
            backend="pydantic",
        )
        request_id_value = request_id(request)
        candidate_count = max(request.top_k, self.config.candidate_count)
        logger.info(
            "[%s] search started query_type=%s top_k=%d candidates=%d",
            request_id_value,
            request.query_type.value,
            request.top_k,
            candidate_count,
        )
        # ------------------------------------------------------------------
        # Candidate acquisition
        #
        # The configured architecture determines which mutually exclusive
        # path produces the candidates consumed by the common refinement and
        # materialization stages below:
        #
        # - temporal: localize cumulative hints into ranked scenes, then pick
        #   one canonical representative frame from each scene;
        # - legacy: retrieve and optionally rerank frame candidates directly.
        # ------------------------------------------------------------------
        progressive = None
        progressive_trace = None
        if self.temporal_core is not None:
            # Temporal path: hints -> SceneCandidate[] -> representative frames.
            progressive_timer = StageTimer(PipelineStage.LOCALIZATION.value)
            progressive = self.temporal_core.localize(
                request.query,
                search_id=request.search_id,
                filters=request.filters,
                task_type=request.query_type,
            )
            budgets = progressive.diagnostics
            progressive_trace = progressive_timer.finish(
                input_count=1,
                output_count=len(progressive.scenes),
                backend=(
                    f"snapshot_diff:{progressive.diff.mode.value};"
                    f"candidate_pool_size={budgets['candidate_pool_size']};"
                    f"top_m_evidence={budgets['top_m_evidence']};"
                    f"scene_top_p_global={budgets['scene_top_p_global']}"
                ),
            )
            candidates = _representative_candidates(progressive.scenes)
            retrieval_result = RetrievalResult(
                candidates=candidates,
                warnings=list(progressive.warnings),
                trace=progressive.trace,
            )
            reranking_ms = 0
        else:
            # Legacy path: direct retrieval -> optional bounded reranking.
            retrieval_result, reranking_ms = rank_candidates(
                request,
                self.retrieval,
                self.reranking,
                candidate_count=candidate_count,
                rerank_count=self.config.rerank_count,
                request_id=request_id_value,
            )
            candidates = retrieval_result.candidates

        # ------------------------------------------------------------------
        # Shared post-acquisition processing
        # ------------------------------------------------------------------
        refinement_started = perf_counter()
        if self.task_type is TaskType.KIS and progressive is None:
            candidates = shape_kis_candidates(
                candidates,
                self.data,
                KISRankingConfig(
                    temporal_window_ms=self.config.temporal_window_ms,
                ),
                minimum_results=request.top_k,
            )
            retrieval_result = retrieval_result.model_copy(
                update={"candidates": candidates}
            )
        refinement_ms = elapsed_ms(refinement_started)
        materialization_started = perf_counter()
        materialization_timer = StageTimer(PipelineStage.MATERIALIZATION.value)
        logger.info(
            "[%s] materialization started selected=%d",
            request_id_value,
            min(request.top_k, len(candidates)),
        )
        response_request = request.model_copy(update={
            "search_id": (
                progressive.search_id
                if progressive is not None
                else request.search_id
            ),
        })
        response = self.materializer.build_response(
            response_request, candidates[: request.top_k], request_id_value
        )
        materialization_trace = materialization_timer.finish(
            input_count=min(request.top_k, len(candidates)),
            output_count=response.total_results,
            backend="canonical_frame_store",
        )
        trace = retrieval_result.trace
        response_stages = [parse_trace]
        if progressive_trace is not None:
            response_stages.append(progressive_trace)
        response_stages.append(materialization_trace)
        for stage in response_stages:
            trace = trace.merged(RetrievalTrace(stages={stage.stage: stage}))
            log_stage(
                logger,
                request_id=request_id_value,
                task_type=request.query_type,
                trace=stage,
            )
        has_index_search = any(
            name in {"search", "index_search"}
            or name.endswith(".search")
            or name.endswith(".index_search")
            for name in trace.stages
        )
        candidate_retrieval_ms = trace.duration_for(
            "search" if has_index_search else "retrieval"
        )
        total_ms = elapsed_ms(started)
        latency = response.latency_ms.model_copy(
            update={
                "query_encoding": int(trace.duration_for("encode")),
                "candidate_retrieval": int(candidate_retrieval_ms),
                "fusion": int(trace.duration_for("fusion")),
                "reranking": int(trace.duration_for("rerank")) or reranking_ms,
                "temporal_refinement": refinement_ms,
                "materialization": elapsed_ms(materialization_started),
                "time_to_first_candidate": int(
                    retrieval_result.time_to_first_candidate_ms or 0
                ),
                "time_to_first_submission": total_ms,
                "total": total_ms,
            }
        )
        response = response.model_copy(
            update={
                "latency_ms": latency,
                "warnings": [
                    *response.warnings,
                    *retrieval_result.warnings,
                ],
                "trace": trace,
            }
        )
        logger.info(
            "[%s] search completed results=%d",
            request_id_value,
            response.total_results,
        )
        return response


def _representative_candidates(scenes) -> list[RetrievalCandidate]:
    """KIS-only head: select one canonical representative after scene ranking."""

    selected: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for scene in scenes:
        if not scene.evidence:
            continue
        midpoint = (scene.start_ms + scene.end_ms) // 2
        evidence = max(
            scene.evidence,
            key=lambda item: (
                item.score,
                -abs(item.frame.timestamp_ms - midpoint),
                -item.frame.frame_idx,
            ),
        )
        if evidence.frame.frame_id in seen:
            continue
        seen.add(evidence.frame.frame_id)
        selected.append(RetrievalCandidate(
            frame_id=evidence.frame.frame_id,
            source_scores=dict(evidence.source_scores),
            source_ranks=dict(evidence.source_ranks),
            fusion_score=evidence.score,
            final_score=scene.final_score,
            metadata={
                "scene_id": scene.scene_id,
                "scene_scores": {
                    "semantic": scene.semantic_score,
                    "coverage": scene.coverage_score,
                    "evaluation_coverage": scene.evaluation_coverage_score,
                    "temporal": scene.temporal_score,
                    "relation": scene.relation_score,
                },
            },
        ))
    return selected
