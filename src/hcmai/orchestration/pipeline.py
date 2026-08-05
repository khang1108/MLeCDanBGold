"""Public orchestration service for online competition search."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    QuerySuggestionRequest,
    QuerySuggestionResponse,
    RetrievalSource,
    SearchRequest,
    SearchResponse,
    SubmissionResult,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.llm.pipeline import LLMService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.ranking import elapsed_ms, rank_candidates, request_id
from hcmai.query_suggestions.pipeline import SuggestionService
from hcmai.reranking.pipeline import RerankingService
from hcmai.retriever.pipeline import RetrievalService

logger = get_logger(__name__)

class UnsupportedSearchTaskError(ValueError):
    """A request cannot be handled by the search application boundary."""


class SearchPipelineUnavailableError(RuntimeError):
    """A known competition task has no executable pipeline yet."""


class SearchServiceUnavailableError(RuntimeError):
    """A required configured search dependency is unavailable."""


class SearchService:
    """Route task requests through the configured capability services."""

    def __init__(
        self,
        data: DataService | None,
        retrieval: RetrievalService | None,
        reranking: RerankingService | None = None,
        config: SearchConfig | None = None,
        suggestion_service: SuggestionService | None = None,
        llm: LLMService | None = None,
    ) -> None:
        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config or SearchConfig()
        self.suggestion_service = suggestion_service
        self.llm = llm
        self.materializer = SearchMaterializer(data) if data is not None else None

    @classmethod
    def load(cls, messages: list[str]) -> SearchService:
        from hcmai.orchestration.setup import load_search_service

        return load_search_service(messages)

    def get_frame(self, frame_id: str) -> FrameRecord:
        if self.data is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return self.data.get_frame(frame_id)

    def neighbors(
        self, frame_id: str, window_ms: int, include_self: bool = True
    ) -> list[FrameRecord]:
        if self.data is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return self.data.neighbors(
            frame_id, window_ms=window_ms, include_self=include_self
        )

    def submission(self, frame_id: str) -> SubmissionResult:
        frame = self.get_frame(frame_id)
        return SubmissionResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            submission_code=f"{frame.video_id},{frame.frame_idx}",
        )

    def suggest(self, request: QuerySuggestionRequest) -> QuerySuggestionResponse:
        if self.suggestion_service is None:
            raise SearchServiceUnavailableError(
                "Query-suggestion provider is not configured"
            )
        return self.suggestion_service.suggest(request)

    def health(self, startup_messages: Sequence[str] = ()) -> dict[str, Any]:
        data_ready = self.data is not None
        retrieval_ready = self.retrieval is not None
        suggestions_ready = self.suggestion_service is not None
        return {
            "status": "ok",
            "ready": data_ready and retrieval_ready,
            "frame_store_loaded": data_ready,
            "retriever_loaded": retrieval_ready,
            "total_frames": self.data.record_count if self.data is not None else 0,
            "evidence_stores": {
                source.value: self.data.has_evidence(source) if self.data else False
                for source in (
                    RetrievalSource.CAPTION,
                    RetrievalSource.OCR,
                    RetrievalSource.ASR,
                )
            },
            "capabilities": {
                "search": retrieval_ready,
                "query_suggestions": {
                    "enabled": suggestions_ready,
                    "provider": (
                        self.suggestion_service.provider_name
                        if self.suggestion_service else None
                    ),
                },
                "frame_assets": data_ready,
                "query_types": {
                    TaskType.KIS.value: retrieval_ready,
                    TaskType.VKIS.value: retrieval_ready,
                    TaskType.VQA.value: False,
                    TaskType.TRAKE.value: False,
                },
            },
            "startup_messages": list(startup_messages),
        }

    def close(self) -> None:
        if self.suggestion_service is not None:
            self.suggestion_service.close()
        if self.llm is not None:
            self.llm.close()

    def search(self, request: SearchRequest) -> SearchResponse:
        self._validate_task(request.query_type)
        if self.data is None or self.materializer is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        if self.retrieval is None:
            raise SearchServiceUnavailableError("Retriever not loaded")
        started = perf_counter()
        request_id_value = request_id(request)
        candidate_count = max(request.top_k, self.config.candidate_count)
        logger.info(
            "[%s] search started query_type=%s top_k=%d candidates=%d",
            request_id_value, request.query_type.value, request.top_k, candidate_count,
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
            request_id_value, min(request.top_k, len(candidates)),
        )
        response = self.materializer.build_response(
            request, candidates[: request.top_k], request_id_value
        )
        latency = response.latency_ms.model_copy(update={
            "candidate_retrieval": retrieval_ms,
            "reranking": reranking_ms,
            "materialization": elapsed_ms(materialization_started),
            "total": elapsed_ms(started),
        })
        response = response.model_copy(update={"latency_ms": latency})
        logger.info(
            "[%s] search completed results=%d",
            request_id_value,
            response.total_results,
        )
        return response

    @staticmethod
    def _validate_task(query_type: TaskType) -> None:
        if query_type is TaskType.VQA:
            # VQA: retrieval -> evidence selection -> answer -> frame binding.
            raise SearchPipelineUnavailableError(
                "pipeline for query_type 'vqa' is not available"
            )
        if query_type is TaskType.TRAKE:
            # TRAKE: event retrieval -> same-video joint temporal alignment.
            raise SearchPipelineUnavailableError(
                "pipeline for query_type 'trake' is not available"
            )
        if query_type not in {TaskType.KIS, TaskType.VKIS, TaskType.KISC}:
            message = f"query_type {query_type.value!r} is not supported"
            raise UnsupportedSearchTaskError(message)
