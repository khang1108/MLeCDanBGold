"""Current frame-search behavior exposed as an executable task pipeline."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from hcmai.common.config import SearchConfig
from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer, log_stage
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalTrace,
    SearchRequest,
    SearchResponse,
    StageStatus,
    StageTrace,
    TaskRequest,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.retrieval.reranking.pipeline import RerankingError, RerankingService
from hcmai.temporal import TemporalEvidenceCore

logger = get_logger(__name__)


def request_id(request: SearchRequest) -> str:
    """Build a unique identifier for one pipeline invocation."""
    return f"request-{uuid4().hex}"


def elapsed_ms(started: float) -> int:
    """Calculate elapsed milliseconds from a perf_counter start timestamp."""
    return max(0, int((perf_counter() - started) * 1_000))


class KISPipeline:
    """Adapt the existing KIS ranking and materialization behavior."""

    def __init__(
        self,
        task_type: TaskType,
        data: DataService | None,
        retrieval: RetrievalService | None,
        config: SearchConfig,
        temporal_core: TemporalEvidenceCore | None = None,
        reranking: RerankingService | None = None,
    ) -> None:
        """Initialize the KIS head and its shared localization dependency."""

        if task_type not in {TaskType.KIS, TaskType.VKIS}:
            raise ValueError(f"KISPipeline cannot handle {task_type.value!r}")
        self._task_type = task_type
        self.data = data
        self.retrieval = retrieval
        self.config = config
        self.temporal_core = temporal_core
        self.reranking = reranking
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
        if self.temporal_core is None:
            raise TaskPipelineDependencyError("Temporal core not loaded")

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
        # Candidate acquisition via Temporal Localization
        # ------------------------------------------------------------------
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
            time_to_first_candidate_ms=progressive.time_to_first_candidate_ms,
        )

        candidates, reranking_trace, reranking_warnings = self._rerank(
            request.query,
            candidates,
            request_id_value,
            request.query_type,
        )
        retrieval_result = retrieval_result.model_copy(
            update={
                "candidates": candidates,
                "warnings": [
                    *retrieval_result.warnings,
                    *reranking_warnings,
                ],
                "trace": (
                    retrieval_result.trace.merged(
                        RetrievalTrace(
                            stages={reranking_trace.stage: reranking_trace}
                        )
                    )
                    if reranking_trace is not None
                    else retrieval_result.trace
                ),
            }
        )

        materialization_started = perf_counter()
        materialization_timer = StageTimer(PipelineStage.MATERIALIZATION.value)

        logger.info(
            "[%s] materialization started selected=%d",
            request_id_value,
            min(request.top_k, len(candidates)),
        )
        response_request = request.model_copy(update={
            "search_id": progressive.search_id,
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
        response_stages = [parse_trace, progressive_trace, materialization_trace]

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
                "reranking": int(trace.duration_for("rerank")),
                "temporal_refinement": 0,
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

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        request_id_value: str,
        task_type: TaskType,
    ) -> tuple[list[RetrievalCandidate], StageTrace | None, list[str]]:
        """Apply bounded visual reranking while preserving temporal fallback.

        Temporal localization remains the source of the candidate pool. The
        reranker may only score and reorder that bounded pool; it cannot create
        or rewrite canonical frame identities. A configured optional reranker
        failure therefore returns the original scene order with a warning.
        """

        if self.reranking is None:
            logger.info(
                "[%s] reranking skipped reason=not_configured",
                request_id_value,
            )
            return candidates, None, []

        if self.config.rerank_count <= 0:
            skipped = StageTimer(PipelineStage.RERANK.value).finish(
                status=StageStatus.SKIPPED,
                attempt_count=0,
                input_count=len(candidates),
                output_count=len(candidates),
                backend=_reranker_backend(self.reranking),
            )
            log_stage(
                logger,
                request_id=request_id_value,
                task_type=task_type,
                trace=skipped,
            )
            logger.info(
                "[%s] reranking skipped reason=disabled",
                request_id_value,
            )
            return candidates, skipped, []

        depth = min(self.config.rerank_count, len(candidates))
        reranking_timer = StageTimer(PipelineStage.RERANK.value)
        logger.info(
            "[%s] reranking started candidates=%d",
            request_id_value,
            depth,
        )
        try:
            ranked = list(self.reranking.rerank(query, candidates[:depth]))
        except RerankingError as error:
            cause = error.__cause__
            reranking_trace = reranking_timer.finish(
                status=StageStatus.PARTIAL,
                error_category=error.category,
                input_count=depth,
                output_count=len(candidates),
                backend=_reranker_backend(self.reranking),
                fallback_used=True,
            )
            log_stage(
                logger,
                request_id=request_id_value,
                task_type=task_type,
                trace=reranking_trace,
            )
            logger.warning(
                "[%s] reranking fallback category=%s candidates=%d "
                "cause_type=%s cause=%s",
                request_id_value,
                error.category,
                depth,
                type(cause).__name__ if cause is not None else "none",
                cause or "none",
            )
            if self.reranking.config.required:
                raise
            return (
                candidates,
                reranking_trace,
                [f"reranking fallback ({error.category})"],
            )

        ranked.extend(candidates[depth:])
        reranking_trace = reranking_timer.finish(
            input_count=depth,
            output_count=len(ranked),
            backend=_reranker_backend(self.reranking),
        )
        log_stage(
            logger,
            request_id=request_id_value,
            task_type=task_type,
            trace=reranking_trace,
        )
        logger.info(
            "[%s] reranking completed candidates=%d",
            request_id_value,
            depth,
        )
        return ranked, reranking_trace, []


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
        scene_frame_ids = [item.frame.frame_id for item in scene.evidence]

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
                "frame_ids": scene_frame_ids,
            },
        ))
    return selected


def _reranker_backend(reranking: RerankingService) -> str:
    """Return the concrete adapter name for request-scoped stage telemetry."""

    adapter = getattr(reranking, "adapter", reranking)
    return type(adapter).__name__
