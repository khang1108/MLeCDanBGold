"""TRAKE key-event alignment exposed as an executable task pipeline."""

from __future__ import annotations

from hashlib import sha1

from hcmai.common.schemas import (
    TaskRequest,
    TaskType,
    TRAKERequest,
    TRAKEResponse,
    TRAKESubmission,
)
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.pipelines.base import TaskPipelineDependencyError
from hcmai.retriever.pipeline import RetrievalService
from hcmai.trake import TRAKESettings, rank_paths

logger = get_logger(__name__)


class TRAKEPipeline:
    """Shortlist videos, align events monotonically, and rank the rows."""

    task_type = TaskType.TRAKE

    def __init__(
        self,
        retrieval: RetrievalService | None,
        settings: TRAKESettings | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.settings = settings or TRAKESettings()

    def execute(self, request: TaskRequest) -> TRAKEResponse:
        if not isinstance(request, TRAKERequest):
            raise ValueError("TRAKEPipeline requires a TRAKE request")
        if self.retrieval is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        events = request.events
        if events is None:
            raise ValueError("TRAKE needs 'events' with at least two ordered events")
        digest = sha1(f"trake\0{request.query}\0{request.top_k}".encode())
        request_id = f"trake-{digest.hexdigest()[:12]}"
        videos = self.retrieval.score_visual_videos(
            events,
            self.settings.top_k,
            self.settings.max_videos,
            self.settings.rrf_k,
            self.settings.chunk_size,
        )
        rows = rank_paths(
            videos,
            self.settings.lambda_gap,
            request.top_k,
            self.settings.event_power,
        )
        logger.info(
            "[%s] trake completed events=%d videos=%d rows=%d",
            request_id,
            len(events),
            len(videos),
            len(rows),
        )
        return TRAKEResponse(
            request_id=request_id,
            query=request.query,
            events=events,
            top_k=request.top_k,
            total_results=len(rows),
            submissions=[
                TRAKESubmission(
                    rank=rank,
                    video_id=row.video_id,
                    frame_ids=list(row.frame_ids),
                    frame_idxs=list(row.frame_idx),
                )
                for rank, row in enumerate(rows, start=1)
            ],
        )
