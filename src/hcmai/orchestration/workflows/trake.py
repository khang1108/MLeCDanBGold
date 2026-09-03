"""Project shared temporal-search paths into the TRAKE HTTP response.

This workflow receives caller-provided ordered events and delegates their
retrieval and monotonic alignment to ``TemporalSearchService``. It does not
split KIS queries, create submissions, merge equal-video paths, or alter
canonical alignment identity.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from hcmai.api.contracts import (
    SearchLatency,
    TRAKEPath,
    TRAKERequest,
    TRAKEResponse,
)
from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.temporal import AlignedPath

if TYPE_CHECKING:
    from hcmai.query_preparation.service import QueryPreparationService

logger = get_logger(__name__)


class TRAKEPipeline:
    """Expose every ranked temporal path as an independent TRAKE result."""

    def __init__(
        self,
        temporal: TemporalSearchService | None,
        query_preparation: QueryPreparationService | None = None,
        max_temporal_event_count: int = DEFAULT_MAX_TEMPORAL_EVENT_COUNT,
    ) -> None:
        """Bind the shared temporal-search service used by this task head."""

        self.temporal = temporal
        self.query_preparation = query_preparation
        self.max_temporal_event_count = max_temporal_event_count

    def execute(self, request: TRAKERequest) -> TRAKEResponse:
        """Align explicit events and project paths without video-level merging.

        Each aligned path retains its raw DP score and canonical frame arrays.
        Asset URLs are intentionally omitted because clients fetch images from
        the single keyframe endpoint using the returned canonical frame IDs.
        """

        if self.temporal is None:
            raise RuntimeError("temporal search service is not loaded")

        started = perf_counter()

        query_started = perf_counter()
        events = request.events
        if len(events) > self.max_temporal_event_count:
            raise ValueError(
                f"requests may contain at most {self.max_temporal_event_count} temporal events"
            )
        retrieval_events = request.retrieval_events or events
        # Candidate rewrites belong only to Dense retrieval. BM25 searches the
        # original Vietnamese events against the Vietnamese context corpus.
        caption_events = events if request.use_bm25 else None
        query_ms = (perf_counter() - query_started) * 1_000

        search = self.temporal.search(
            events,
            retrieval_events=retrieval_events,
            caption_events=caption_events,
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            top_k=request.top_k,
        )

        materialization_started = perf_counter()
        paths = [self._build_path(path) for path in search.paths]
        materialization_ms = (perf_counter() - materialization_started) * 1_000
        total_ms = (perf_counter() - started) * 1_000

        logger.info(
            "TRAKE completed events=%d paths=%d videos=%d",
            len(events),
            len(paths),
            len({path.video_id for path in paths}),
        )

        return TRAKEResponse(
            events=list(events),
            dense_events=list(retrieval_events) if request.use_dense else None,
            bm25_caption_events=(list(caption_events) if caption_events else None),
            use_dense=request.use_dense,
            use_bm25=request.use_bm25,
            paths=paths,
            latency=SearchLatency(
                query_ms=query_ms,
                retrieval_ms=search.retrieval_ms,
                alignment_ms=search.alignment_ms,
                materialization_ms=materialization_ms,
                total_ms=total_ms,
        ),)

    @staticmethod
    def _build_path(path: AlignedPath) -> TRAKEPath:
        """Convert one canonical aligned path without changing its coordinates."""
        frame_ids = list(path.frame_ids)
        return TRAKEPath(
            video_id=path.video_id,
            score=path.score,
            frame_ids=frame_ids,
            frame_idxs=list(path.frame_idxs),
            timestamps_ms=list(path.timestamps_ms),
        )
