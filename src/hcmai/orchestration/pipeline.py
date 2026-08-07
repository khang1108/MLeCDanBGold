"""Public orchestration service for online competition search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    QuerySuggestionRequest,
    QuerySuggestionResponse,
    RetrievalSource,
    SubmissionResult,
    TaskRequest,
    TaskResponse,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.llm.pipeline import LLMService
from hcmai.orchestration.pipelines.base import TaskPipelineDependencyError
from hcmai.orchestration.pipelines.kis import KISPipeline
from hcmai.orchestration.pipelines.trake import TRAKEPipeline
from hcmai.orchestration.task_router import PipelineRegistry
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
        pipeline_registry: PipelineRegistry | None = None,
    ) -> None:
        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config or SearchConfig()
        self.suggestion_service = suggestion_service
        self.llm = llm
        self.pipeline_registry = (
            pipeline_registry
            if pipeline_registry is not None
            else self._default_registry()
        )

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
        task_capabilities = self.pipeline_registry.capability_report(
            (TaskType.KIS, TaskType.VKIS, TaskType.VQA, TaskType.TRAKE)
        )
        task_capabilities = {
            task_type: registered and data_ready and retrieval_ready
            for task_type, registered in task_capabilities.items()
        }
        search_ready = any(task_capabilities.values())
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
                "search": search_ready,
                "query_suggestions": {
                    "enabled": suggestions_ready,
                    "provider": (
                        self.suggestion_service.provider_name
                        if self.suggestion_service else None
                    ),
                },
                "frame_assets": data_ready,
                "query_types": task_capabilities,
            },
            "startup_messages": list(startup_messages),
        }

    def close(self) -> None:
        if self.suggestion_service is not None:
            self.suggestion_service.close()
        if self.llm is not None:
            self.llm.close()

    def search(self, request: TaskRequest) -> TaskResponse:
        try:
            pipeline = self.pipeline_registry.get(request.query_type)
        except KeyError as error:
            raise SearchPipelineUnavailableError(
                f"pipeline for query_type {request.query_type.value!r} "
                "is not available"
            ) from error
        try:
            return pipeline.execute(request)
        except TaskPipelineDependencyError as error:
            raise SearchServiceUnavailableError(str(error)) from error
        except ValueError as error:
            raise UnsupportedSearchTaskError(str(error)) from error

    def _default_registry(self) -> PipelineRegistry:
        task_types = (TaskType.KIS, TaskType.VKIS, TaskType.KISC)
        registry = PipelineRegistry(
            KISPipeline(
                task_type,
                self.data,
                self.retrieval,
                self.reranking,
                self.config,
            )
            for task_type in task_types
        )
        registry.register(TRAKEPipeline(self.retrieval))
        return registry
