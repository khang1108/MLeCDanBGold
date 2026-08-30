"""KIS projection of stateless ordered event-to-frame alignment paths.

This task head turns a shared canonical alignment path into one competition
frame result. It does not run retrieval, reranking, progressive state, or
scene clustering; those concerns are deliberately outside the KIS boundary.
"""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from hcmai.common.observability import PipelineStage
from hcmai.common.observability.tracing import StageTimer, log_stage
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalTrace,
    SearchRequest,
    SearchResponse,
    TaskRequest,
    TaskType,
)
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
from hcmai.orchestration.materializer import SearchMaterializer
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.orchestration.workflows.base import (
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.temporal import AlignedPath, split_query_events

logger = get_logger(__name__)


def request_id(request: SearchRequest) -> str:
    """Build a unique identifier for one stateless KIS pipeline invocation."""

    del request
    return f"request-{uuid4().hex}"


def elapsed_ms(started: float) -> int:
    """Calculate elapsed milliseconds from a ``perf_counter`` start value."""

    return max(0, int((perf_counter() - started) * 1_000))


class KISPipeline:
    """Return one deterministic representative frame for each aligned path."""

    task_type = TaskType.KIS

    def __init__(
        self,
        data: DataService | None,
        alignment: TemporalSearchService | None,
    ) -> None:
        """Initialize canonical materialization and the shared alignment facade."""

        self.data = data
        self.alignment = alignment
        self.materializer = SearchMaterializer(data) if data is not None else None

    def execute(self, request: TaskRequest) -> SearchResponse:
        """Align query events and materialize the midpoint of each path.

        One-event paths return their only frame; multi-event paths return the
        upper midpoint. The complete path remains in ``SearchResult.frame_ids``
        so the task response retains temporal evidence.
        """

        if not isinstance(request, SearchRequest):
            raise TaskPipelineRequestError(
                "KISPipeline requires a search request"
            )
        if request.query_type is not self.task_type:
            raise TaskPipelineRequestError(
                f"pipeline for {self.task_type.value!r} cannot execute "
                f"request for {request.query_type.value!r}"
            )
        if self.data is None or self.materializer is None:
            raise TaskPipelineDependencyError("Frame store not loaded")
        if self.alignment is None:
            raise TaskPipelineDependencyError("Alignment service not loaded")
        if request.filters is not None and request.filters.min_score is not None:
            raise TaskPipelineRequestError(
                "min_score is not supported for multi-event alignment; "
                "use video/time filters"
            )

        started = perf_counter()
        request_id_value = request_id(request)
        parse_trace = StageTimer(PipelineStage.PARSE.value).finish(
            input_count=1,
            output_count=1,
            backend="pydantic",
        )
        events = split_query_events(request.query)

        alignment_timer = StageTimer(PipelineStage.LOCALIZATION.value)
        search = self.alignment.search(
            events,
            top_k=request.top_k,
        )
        alignment_trace = alignment_timer.finish(
            input_count=len(events),
            output_count=len(search.paths),
            backend="monotonic_dp",
        )
        candidates = [
            _path_to_candidate(path, self.data) for path in search.paths
        ]

        materialization_started = perf_counter()
        materialization_timer = StageTimer(PipelineStage.MATERIALIZATION.value)
        response_search_id = request.search_id or f"search-{uuid4().hex}"
        response_request = request.model_copy(
            update={"search_id": response_search_id}
        )
        response = self.materializer.build_response(
            response_request,
            candidates,
            request_id_value,
        )
        materialization_trace = materialization_timer.finish(
            input_count=len(candidates),
            output_count=response.total_results,
            backend="canonical_frame_store",
        )

        trace = RetrievalTrace()
        for stage in (parse_trace, alignment_trace, materialization_trace):
            trace = trace.merged(RetrievalTrace(stages={stage.stage: stage}))
            log_stage(
                logger,
                request_id=request_id_value,
                task_type=self.task_type,
                trace=stage,
            )

        total_ms = elapsed_ms(started)
        response = response.model_copy(
            update={
                "latency_ms": response.latency_ms.model_copy(
                    update={
                        "temporal_refinement": int(alignment_trace.duration_ms),
                        "materialization": elapsed_ms(materialization_started),
                        "time_to_first_candidate": (
                            int(alignment_trace.duration_ms) if candidates else 0
                        ),
                        "time_to_first_submission": total_ms,
                        "total": total_ms,
                    }
                ),
                "trace": trace,
            }
        )
        logger.info(
            "[%s] KIS completed events=%d videos=%d paths=%d",
            request_id_value,
            len(events),
            len({path.video_id for path in search.paths}),
            len(candidates),
        )
        return response


def _path_to_candidate(path: AlignedPath, data: DataService) -> RetrievalCandidate:
    """Select a midpoint frame while retaining the complete canonical path."""

    frame_ids = path.frame_ids
    frame = data.get_frame(frame_ids[len(frame_ids) // 2])
    return RetrievalCandidate(
        frame_id=frame.frame_id,
        final_score=path.score,
        metadata={
            "frame_ids": list(frame_ids),
        },
    )
