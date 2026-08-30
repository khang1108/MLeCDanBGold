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
from hcmai.data.pipeline import DataService
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)

logger = get_logger(__name__)


class TRAKEPipeline:
    """Project shared ordered alignment paths into TRAKE submissions."""

    task_type = TaskType.TRAKE

    def __init__(
        self,
        data: DataService | None,
        alignment: TemporalSearchService | None,
    ) -> None:
        """Initialize canonical data access and the shared alignment facade."""

        self.data = data
        self.alignment = alignment

    def execute(self, request: TaskRequest) -> TRAKEResponse:
        """Align ordered events and project canonical IDs into a TRAKE response."""

        if not isinstance(request, TRAKERequest):
            raise TaskPipelineRequestError(
                "TRAKEPipeline requires a TRAKE request"
            )

        if self.alignment is None:
            raise TaskPipelineDependencyError("Alignment service not loaded")
        if self.data is None:
            raise TaskPipelineDependencyError("Frame store not loaded")

        events = request.events
        if events is None:
            raise TaskPipelineRequestError(
                "TRAKE needs 'events' with at least two ordered events"
            )

        digest = sha1(f"trake\0{request.query}\0{request.top_k}".encode())

        request_id = f"trake-{digest.hexdigest()[:12]}"

        search = self.alignment.search(events, top_k=request.top_k)
        rows = search.paths

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
                        fallback_path=(
                            self.data.get_frame(row.frame_ids[0]).image_path
                            if row.frame_ids
                            else None
                        ),
                    ),
                    frame_ids=list(row.frame_ids),
                    frame_idxs=list(row.frame_idxs),
                    timestamps_ms=list(row.timestamps_ms),
                    fps=derive_fps(
                        self.data.get_frame(row.frame_ids[0])
                        if row.frame_ids
                        else None
                    ),
                )
                for rank, row in enumerate(rows, start=1)
            ],
        )
