"""Public orchestration service for online competition search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from hcmai.common.config import SearchConfig, VQAConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalSource,
    SubmissionResult,
    TaskRequest,
    TaskResponse,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.llm.pipeline import LLMService
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.pipelines.vqa.pipeline import VQAPipeline
from hcmai.orchestration.task_router import PipelineRegistry
from hcmai.retrieval.reranking.pipeline import RerankingService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.common.observability import METRICS
from hcmai.temporal import TemporalEvidenceCore

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
        llm: LLMService | None = None,
        vqa_config: VQAConfig | None = None,
        pipeline_registry: PipelineRegistry | None = None,
    ) -> None:
        """Initialize task pipelines from configured capability services."""

        self.data = data
        self.retrieval = retrieval
        self.reranking = reranking
        self.config = config or SearchConfig()
        self.llm = llm
        self.vqa_config = vqa_config or VQAConfig()
        self.pipeline_registry = (
            pipeline_registry
            if pipeline_registry is not None
            else self._default_registry()
        )

    @classmethod
    def load(cls, messages: list[str]) -> SearchService:
        """Load the configured search service and append startup diagnostics."""

        from hcmai.orchestration.setup import load_search_service

        return load_search_service(messages)

    def get_frame(self, frame_id: str) -> FrameRecord:
        """Resolve one frame through the canonical data authority."""

        if self.data is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return self.data.get_frame(frame_id)

    def neighbors(
        self, frame_id: str, window_ms: int, include_self: bool = True
    ) -> list[FrameRecord]:
        """Return canonical temporal neighbors around one frame."""

        if self.data is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return self.data.neighbors(
            frame_id, window_ms=window_ms, include_self=include_self
        )

    def submission(self, frame_id: str) -> SubmissionResult:
        """Build the official submission identity for one canonical frame."""

        frame = self.get_frame(frame_id)
        return SubmissionResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            submission_code=f"{frame.video_id},{frame.frame_idx}",
        )

    def health(self, startup_messages: Sequence[str] = ()) -> dict[str, Any]:
        """Report readiness and capability status without mutating services."""

        data_ready = self.data is not None
        retrieval_ready = self.retrieval is not None
        asset_status = self._frame_asset_status()
        active_sources = (
            set(getattr(self.retrieval, "active_sources", (RetrievalSource.VISUAL,)))
            if self.retrieval is not None
            else set()
        )
        task_capabilities = self.pipeline_registry.capability_report(
            (TaskType.KIS, TaskType.VKIS, TaskType.VQA, TaskType.TRAKE)
        )
        task_capabilities = {
            task_type: registered and data_ready and retrieval_ready
            for task_type, registered in task_capabilities.items()
        }
        search_ready = any(task_capabilities.values())
        default_remote_capabilities = {
            "embedding": False,
            "reranking": False,
            "multi_image_vqa": False,
            "structured_parsing": False,
        }
        capability_health = (
            getattr(self.llm, "capability_health", None)
            if self.llm is not None
            else None
        )
        remote_capabilities = (
            capability_health()
            if capability_health is not None
            else default_remote_capabilities
        )
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
            "remote_inference": (
                self.llm.gateway_health()
                if self.llm is not None
                else {
                    "configured": False,
                    "circuit_state": "not_configured",
                }
            ),
            "retrieval_modalities": {
                source.value: {
                    "active": source in active_sources,
                    "required": source in self.config.fusion.required_sources,
                }
                for source in RetrievalSource
            },
            "observability": METRICS.snapshot(),
            "capabilities": {
                "search": search_ready,
                "kis": task_capabilities.get(TaskType.KIS.value, False),
                "vqa": task_capabilities.get(TaskType.VQA.value, False),
                "shared_retrieval": retrieval_ready,
                "remote_inference": remote_capabilities,
                "frame_assets": asset_status["ready"],
                "frame_asset_status": asset_status,
                "query_types": task_capabilities,
            },
            "startup_messages": list(startup_messages),
        }

    def _frame_asset_status(self) -> dict[str, int | bool]:
        """Return bounded frame-asset readiness diagnostics."""

        if not isinstance(self.data, DataService):
            return {"ready": False, "checked": 0, "available": 0, "missing": 0}
        return self.data.frame_asset_status().as_dict()

    def close(self) -> None:
        """Close optional inference resources owned by the service."""

        if self.llm is not None:
            self.llm.close()

    def search(self, request: TaskRequest) -> TaskResponse:
        """Dispatch a validated task request through its registered pipeline."""

        try:
            pipeline = self.pipeline_registry.get(request.query_type)
        except KeyError as error:
            raise SearchPipelineUnavailableError(
                f"pipeline for query_type {request.query_type.value!r} "
                "is not available"
            ) from error
        try:
            return cast(Any, pipeline).execute(request)
        except TaskPipelineDependencyError as error:
            raise SearchServiceUnavailableError(str(error)) from error
        except TaskPipelineRequestError as error:
            raise UnsupportedSearchTaskError(str(error)) from error

    def _default_registry(self) -> PipelineRegistry:
        """Build task heads and share one temporal core between KIS and VQA."""

        temporal_core = (
            TemporalEvidenceCore(self.data, self.retrieval, self.config)
            if isinstance(self.data, DataService)
            and isinstance(self.retrieval, RetrievalService)
            and self.config.progressive.architecture == "temporal"
            else None
        )
        task_types = (TaskType.KIS, TaskType.VKIS)
        pipelines = [
            KISPipeline(
                task_type,
                self.data,
                self.retrieval,
                self.reranking,
                self.config,
                temporal_core,
            )
            for task_type in task_types
        ]
        pipelines.append(
            cast(
                Any,
                VQAPipeline(
                    self.data,
                    self.retrieval,
                    self.llm,
                    self.vqa_config,
                    temporal_core,
                ),
            )
        )
        pipelines.append(cast(Any, TRAKEPipeline(self.retrieval)))
        return PipelineRegistry(pipelines)
