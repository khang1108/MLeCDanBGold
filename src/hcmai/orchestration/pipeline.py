"""Public orchestration service for online competition search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from hcmai.api.contracts import (
    SearchRequest,
    SearchResponse,
    TRAKERequest,
    TRAKEResponse,
)
from hcmai.common.config import SearchConfig
from hcmai.common.schemas import (
    FrameCatalogEntry,
    FrameRecord,
    RetrievalSource,
    SubmissionResult,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.common.observability import METRICS
from hcmai.common.utils.video import official_frame_idx
from hcmai.orchestration.temporal_search import TemporalSearchService

if TYPE_CHECKING:
    from hcmai.retrieval.retriever.pipeline import RetrievalService
    from thundercompute.pipeline import LLMService

logger = get_logger(__name__)


class SearchServiceUnavailableError(RuntimeError):
    """A required configured search dependency is unavailable."""


class SearchService:
    """Expose explicit KIS and TRAKE workflows over shared runtime services."""

    def __init__(
        self,
        data: DataService | None,
        retrieval: RetrievalService | None,
        config: SearchConfig | None = None,
        llm: LLMService | None = None,
    ) -> None:
        """Initialize explicit task workflows over one temporal service."""

        self.data = data
        self.retrieval = retrieval
        self.config = config or SearchConfig()
        self.llm = llm

        temporal = (
            TemporalSearchService(
                self.data,
                self.retrieval,
                self.config.alignment,
            )
            if self.data is not None and self.retrieval is not None
            else None
        )
        self.kis = KISPipeline(self.data, temporal)
        self.trake = TRAKEPipeline(temporal)

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

    def list_frames(self) -> list[FrameCatalogEntry]:
        """Return every canonical keyframe with its loaded catalog evidence."""

        if self.data is None:
            raise SearchServiceUnavailableError("Frame store not loaded")
        return list(self.data.iter_frame_catalog_entries())

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
        frame_idx = official_frame_idx(frame)
        return SubmissionResult(
            frame_id=frame.frame_id,
            video_id=frame.video_id,
            frame_idx=frame_idx,
            submission_code=f"{frame.video_id},{frame_idx}",
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
        search_ready = data_ready and retrieval_ready
        default_remote_capabilities = {
            "embedding": False,
            "reranking": False,
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
                "kis": search_ready,
                "trake": search_ready,
                "shared_retrieval": retrieval_ready,
                "remote_inference": remote_capabilities,
                "frame_assets": asset_status["ready"],
                "frame_asset_status": asset_status,
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

    def search_kis(self, request: SearchRequest) -> SearchResponse:
        """Execute a validated KIS request through the explicit KIS workflow."""

        self._ensure_search_ready()
        return self.kis.execute(request)

    def search_trake(self, request: TRAKERequest) -> TRAKEResponse:
        """Execute a validated TRAKE request through the explicit TRAKE workflow."""

        self._ensure_search_ready()
        return self.trake.execute(request)

    def _ensure_search_ready(self) -> None:
        """Reject online search when canonical data or retrieval is unavailable."""

        missing: list[str] = []
        if self.data is None:
            missing.append("canonical frame data")
        if self.retrieval is None:
            missing.append("retrieval service")
        if missing:
            raise SearchServiceUnavailableError(
                f"Search dependencies not loaded: {', '.join(missing)}"
            )
