"""Current frame-search behavior exposed as an executable task pipeline."""

from __future__ import annotations

from time import perf_counter

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import SearchRequest, SearchResponse, TaskType
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.pipelines.base import TaskPipelineDependencyError
from hcmai.orchestration.ranking import elapsed_ms, rank_candidates, request_id
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
        if task_type not in {TaskType.KIS, TaskType.VKIS, TaskType.KISC}:
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
        request_id_value = request_id(request)
        candidate_count = max(request.top_k, self.config.candidate_count)
        logger.info(
            "[%s] search started query_type=%s top_k=%d candidates=%d",
            request_id_value,
            request.query_type.value,
            request.top_k,
            candidate_count,
        )
        candidates, retrieval_ms, reranking_ms = rank_candidates(
            request,
            self.retrieval,
            self.reranking,
            candidate_count=candidate_count,
            rerank_count=self.config.rerank_count,
            request_id=request_id_value,
        )
        materialization_started = perf_counter()
        logger.info(
            "[%s] materialization started selected=%d",
            request_id_value,
            min(request.top_k, len(candidates)),
        )
        response = self.materializer.build_response(
            request, candidates[: request.top_k], request_id_value
        )
        latency = response.latency_ms.model_copy(
            update={
                "candidate_retrieval": retrieval_ms,
                "reranking": reranking_ms,
                "materialization": elapsed_ms(materialization_started),
                "total": elapsed_ms(started),
            }
        )
        response = response.model_copy(update={"latency_ms": latency})
        logger.info(
            "[%s] search completed results=%d",
            request_id_value,
            response.total_results,
        )
        return response
