"""TRAKE key-event alignment exposed as an executable task pipeline."""

from __future__ import annotations

from hashlib import sha1

from hcmai.agents.trake import (
    TrakePath,
    event_video_scores,
    rank_paths,
    split_delimited,
)
from hcmai.common.config import SearchConfig
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

logger = get_logger(__name__)


class TRAKEPipeline:
    """Shortlist videos, align events monotonically, and rank the rows."""

    task_type = TaskType.TRAKE

    def __init__(
        self,
        retrieval: RetrievalService | None,
        config: SearchConfig,
        lambda_gap: float = 1e-5,
    ) -> None:
        self.retrieval = retrieval
        self.config = config
        self.lambda_gap = lambda_gap

    def execute(self, request: TaskRequest) -> TRAKEResponse:
        if not isinstance(request, TRAKERequest):
            raise ValueError("TRAKEPipeline requires a TRAKE request")
        if self.retrieval is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        events = request.events or split_delimited(request.query)
        if events is None or len(events) < 2:
            raise ValueError(
                "TRAKE needs at least two ordered events; send 'events' or a "
                "'|'-delimited query"
            )
        request_id = _request_id(request)
        videos = event_video_scores(
            self.retrieval, events, self.config.candidate_count
        )
        rows = rank_paths(videos, self.lambda_gap, request.top_k)
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
                _submission(row, rank) for rank, row in enumerate(rows, start=1)
            ],
        )


def _submission(row: TrakePath, rank: int) -> TRAKESubmission:
    """Materialize one aligned path through its canonical frame identities."""

    return TRAKESubmission(
        rank=rank,
        video_id=row.video_id,
        frame_ids=list(row.frame_ids),
        frame_idxs=list(row.frame_idx),
    )


def _request_id(request: TRAKERequest) -> str:
    payload = f"trake\0{request.query}\0{request.top_k}".encode()
    return f"trake-{sha1(payload).hexdigest()[:12]}"
