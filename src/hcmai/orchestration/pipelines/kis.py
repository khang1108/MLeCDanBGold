"""Current frame-search behavior exposed as an executable task pipeline."""

from __future__ import annotations

from time import perf_counter

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    RetrievalTrace,
    SearchRequest,
    SearchResponse,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.kis.ranking import KISRankingConfig, shape_kis_candidates
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.pipelines.base import TaskPipelineDependencyError
from hcmai.orchestration.ranking import elapsed_ms, rank_candidates, request_id
from hcmai.observability import PipelineStage
from hcmai.observability.tracing import StageTimer, log_stage
from hcmai.reranking.pipeline import RerankingService
from hcmai.retriever.pipeline import RetrievalService

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
    ) -> None:
        if task_type not in {TaskType.KIS, TaskType.VKIS}:
            raise ValueError(f"KISPipeline cannot handle {task_type.value!r}")
        self._task_type = task_type
        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config
        self.materializer = SearchMaterializer(data) if data is not None else None

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def execute(self, request: SearchRequest) -> SearchResponse:
        if request.query_type is not self.task_type:
            raise ValueError(
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
        retrieval_result, reranking_ms = rank_candidates(
            request,
            self.retrieval,
            self.reranking,
            candidate_count=candidate_count,
            rerank_count=self.config.rerank_count,
            request_id=request_id_value,
        )
        candidates = retrieval_result.candidates
        refinement_started = perf_counter()
        if self.task_type is TaskType.KIS:
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
        response = self.materializer.build_response(
            request, candidates[: request.top_k], request_id_value
        )
        materialization_trace = materialization_timer.finish(
            input_count=min(request.top_k, len(candidates)),
            output_count=response.total_results,
            backend="canonical_frame_store",
        )
        trace = retrieval_result.trace
        for stage in (parse_trace, materialization_trace):
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
