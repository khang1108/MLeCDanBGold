"""Project shared temporal-search paths into the TRAKE HTTP response.

This workflow receives caller-provided ordered events and delegates their
retrieval and monotonic alignment to ``TemporalSearchService``. It does not
split KIS queries, create submissions, merge equal-video paths, or alter
canonical alignment identity.
"""

from __future__ import annotations

from time import perf_counter
from hcmai.api.contracts import (
    SearchLatency,
    TRAKEPath,
    TRAKERequest,
    TRAKEResponse,
)
from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.temporal_search import TemporalSearchService
from hcmai.temporal import AlignedPath

logger = get_logger(__name__)


class TRAKEPipeline:
    """Expose every ranked temporal path as an independent TRAKE result."""

    def __init__(
        self,
        temporal: TemporalSearchService | None,
    ) -> None:
        """Bind the shared temporal-search service used by this task head."""

        self.temporal = temporal

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
        query_ms = (perf_counter() - query_started) * 1_000

        search = self.temporal.search(events, top_k=request.top_k)

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
            paths=paths,
            latency=SearchLatency(
                query_ms=query_ms,
                retrieval_ms=search.retrieval_ms,
                alignment_ms=search.alignment_ms,
                materialization_ms=materialization_ms,
                total_ms=total_ms,
            ),
        )

    @classmethod
    def _build_path(cls, path: AlignedPath) -> TRAKEPath:
        """Convert one canonical aligned path without changing its coordinates."""

        frame_ids = list(path.frame_ids)
        return TRAKEPath(
            video_id=path.video_id,
            score=path.score,
            frame_ids=frame_ids,
            frame_idxs=list(path.frame_idxs),
            timestamps_ms=list(path.timestamps_ms),
        )
