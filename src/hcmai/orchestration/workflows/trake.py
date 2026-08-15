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
from hcmai.common.utils.video import derive_fps, format_video_id
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.temporal import TemporalEvidenceCore

logger = get_logger(__name__)


class TRAKEPipeline:
    """Shortlist videos, align events monotonically, and rank the rows."""

    task_type = TaskType.TRAKE

    def __init__(
        self,
        temporal_core: TemporalEvidenceCore | None,
    ) -> None:
        self.temporal_core = temporal_core

    def execute(self, request: TaskRequest) -> TRAKEResponse:
        if not isinstance(request, TRAKERequest):
            raise TaskPipelineRequestError(
                "TRAKEPipeline requires a TRAKE request"
            )
        if self.temporal_core is None:
            raise TaskPipelineDependencyError("Retriever not loaded")

        events = request.events
        if events is None:
            raise TaskPipelineRequestError(
                "TRAKE needs 'events' with at least two ordered events"
            )
        digest = sha1(f"trake\0{request.query}\0{request.top_k}".encode())
        request_id = f"trake-{digest.hexdigest()[:12]}"
        plan = self.temporal_core.ordered_plan(events)
        aligned = self.temporal_core.align_ordered(
            plan,
            max_paths=request.top_k,
        )
        rows = aligned.paths
        logger.info(
            "[%s] trake completed events=%d videos=%d rows=%d",
            request_id,
            len(events),
            len({row.video_id for row in rows}),
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
                    video_id=format_video_id(
                        row.video_id,
                        fallback_path=row.frames[0].image_path if row.frames else None,
                    ),
                    frame_ids=[frame.frame_id for frame in row.frames],
                    frame_idxs=[
                        frame.frame_idx
                        if frame.frame_idx is not None
                        else round(frame.timestamp_ms * derive_fps(frame) / 1000.0)
                        for frame in row.frames
                    ],
                    fps=derive_fps(row.frames[0] if row.frames else None),
                )
                for rank, row in enumerate(rows, start=1)
            ],
        )
