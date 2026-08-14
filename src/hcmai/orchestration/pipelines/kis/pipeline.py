"""Current frame-search behavior exposed as an executable task pipeline."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalTrace,
    SearchRequest,
    SearchResponse,
    StageTrace,
    TaskRequest,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.kis.ranking import KISRankingConfig, shape_kis_candidates
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.pipelines.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.orchestration.pipelines.kis.representative import RepresentativeFrameSelector
from hcmai.orchestration.pipelines.kis.scene_rerank import rerank_scenes
from hcmai.orchestration.ranking import elapsed_ms, rank_candidates, request_id
from hcmai.observability import PipelineStage
from hcmai.observability.tracing import StageTimer, log_stage
from hcmai.reranking.pipeline import RerankingService
from hcmai.retriever.pipeline import RetrievalService
from hcmai.temporal.engine import TemporalEvidenceEngine
from hcmai.temporal.retrieval import SparseEvidenceProvider
from hcmai.temporal.settings import TemporalSettings
from hcmai.temporal.state import ProgressiveStateStore

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
        temporal_settings: TemporalSettings | None = None,
    ) -> None:
        if task_type not in {TaskType.KIS, TaskType.VKIS}:
            raise ValueError(f"KISPipeline cannot handle {task_type.value!r}")
        self._task_type = task_type
        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config
        self.materializer = SearchMaterializer(data) if data is not None else None
        self.representative = RepresentativeFrameSelector()
        self.temporal_settings = temporal_settings or TemporalSettings()
        # VKIS stays on the legacy frame path until it is evaluated separately.
        self.progressive: TemporalEvidenceEngine | None = None
        if (
            config.progressive_scene_enabled
            and task_type is TaskType.KIS
            and data is not None
            and retrieval is not None
        ):
            self.progressive = self.temporal_settings.engine(
                SparseEvidenceProvider(retrieval, data),
                ProgressiveStateStore(
                    ttl_seconds=config.progressive_state_ttl_seconds,
                    max_states=config.progressive_max_states,
                ),
            )

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def execute(self, request: TaskRequest) -> SearchResponse:
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
        if self.progressive is not None:
            return self._scene_response(
                request,
                self.progressive,
                self.materializer,
                started,
                parse_trace,
                request_id_value,
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

    def _scene_response(
        self,
        request: SearchRequest,
        engine: TemporalEvidenceEngine,
        materializer: SearchMaterializer,
        started: float,
        parse_trace: StageTrace,
        request_id_value: str,
    ) -> SearchResponse:
        """Assemble scenes from progressive evidence, then materialize one frame per scene."""
        result = replace(
            engine, top_k=self.config.candidate_count, max_total=request.top_k
        ).search(
            request.query,
            task_type=TaskType.KIS,
            search_id=request.search_id,
            filters=request.filters,
            allow_missing_state_fallback=True,
        )
        if result.commit_required:
            engine.states.commit(result.state, expected_version=result.state.version)
        logger.info(
            "[%s] scenes assembled search_id=%s scenes=%d",
            request_id_value,
            result.search_id,
            len(result.scenes),
        )
        scenes = result.scenes
        warnings = list(result.warnings)
        stages = [parse_trace]
        # The state keeps the engine's own order; reranking only reorders this response.
        if self.config.scene_rerank_enabled and self.reranking is not None:
            reranked = rerank_scenes(
                request.query,
                scenes,
                self.reranking,
                self.representative,
                self.config,
            )
            scenes = reranked.scenes
            warnings.extend(reranked.warnings)
            stages.append(reranked.trace)
            logger.info(
                "[%s] scene reranking probes=%d elapsed_ms=%d",
                request_id_value,
                reranked.trace.input_count,
                int(reranked.trace.duration_ms),
            )

        materialization_started = perf_counter()
        materialization_timer = StageTimer(PipelineStage.MATERIALIZATION.value)
        by_frame: dict[str, RetrievalCandidate] = {}
        for scene in scenes:
            candidate = self.representative.select(scene)
            if candidate is not None:
                by_frame.setdefault(candidate.frame_id, candidate)
        candidates = list(by_frame.values())
        response = materializer.build_response(request, candidates, request_id_value)
        materialization_trace = materialization_timer.finish(
            input_count=len(candidates),
            output_count=response.total_results,
            backend="canonical_frame_store",
        )
        trace = result.trace
        for stage in (*stages, materialization_trace):
            trace = trace.merged(RetrievalTrace(stages={stage.stage: stage}))
            log_stage(
                logger,
                request_id=request_id_value,
                task_type=request.query_type,
                trace=stage,
            )
        total_ms = elapsed_ms(started)
        logger.info(
            "[%s] search completed results=%d",
            request_id_value,
            response.total_results,
        )
        return response.model_copy(
            update={
                "search_id": result.search_id,
                "latency_ms": response.latency_ms.model_copy(
                    update={
                        "candidate_retrieval": int(
                            trace.duration_for("search") or trace.duration_for("retrieval")
                        ),
                        "reranking": int(trace.duration_for("rerank")),
                        "materialization": elapsed_ms(materialization_started),
                        "time_to_first_submission": total_ms,
                        "total": total_ms,
                    }
                ),
                "warnings": [*response.warnings, *warnings],
                "trace": trace,
            }
        )
