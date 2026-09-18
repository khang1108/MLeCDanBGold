"""Direct AVS text retrieval and coverage materialization workflow.

This module coordinates direct text retrieval from the standalone retrieval service,
applies the deterministic AvsCoverageSelector, and materializes canonical frame
metadata without KIS rewriting, temporal DP, or visual reranking.
"""

from __future__ import annotations

from time import perf_counter

from hcmai.api.contracts.avs import (
    AvsSearchLatency,
    AvsSearchRequest,
    AvsSearchResponse,
    AvsSearchResult,
)
from hcmai.common.config import AvsConfig
from hcmai.corpus import Corpus
from hcmai.orchestration.utils.errors import InvalidQueryInputError
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.orchestration.workflows.avs import AvsCoverageCandidate, AvsCoverageSelector
from hcmai.retrieval.serving.client import RetrievalHttpClient


class AvsSearchService:
    def __init__(
        self,
        *,
        corpus: Corpus,
        retrieval: RetrievalHttpClient,
        config: AvsConfig,
    ) -> None:
        self.corpus = corpus
        self.retrieval = retrieval
        self.config = config
        self.selector = AvsCoverageSelector(config)
        self.materializer = SearchMaterializer(corpus)

    def search(self, request: AvsSearchRequest) -> AvsSearchResponse:
        if request.page_size > self.config.maximum_page_size:
            raise InvalidQueryInputError(
                f"page_size must be <= {self.config.maximum_page_size}"
            )

        started = perf_counter()
        remote = self.retrieval.search_text(
            request.query,
            top_k=self.config.candidate_pool_size,
        )

        canonical = []
        for item in remote.candidates:
            frame = self.corpus.frame(item.frame_id)
            canonical.append(AvsCoverageCandidate(
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                timestamp_ms=frame.timestamp_ms,
                retrieval_rank=item.rank,
                retrieval_score=item.score,
            ))

        coverage_started = perf_counter()
        ordered = self.selector.order(canonical)
        coverage_ms = (perf_counter() - coverage_started) * 1000

        materialization_started = perf_counter()
        visible = ordered[:request.page_size]
        results = []
        for candidate in visible:
            frame = self.corpus.frame(candidate.frame_id)
            results.append(AvsSearchResult(
                candidate_id=frame.frame_id,
                frame_id=frame.frame_id,
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                timestamp_ms=frame.timestamp_ms,
                fps=frame.fps,
                retrieval_rank=candidate.retrieval_rank,
                retrieval_score=candidate.retrieval_score,
                metadata=self.materializer.build_frame_metadata(frame),
            ))
        materialization_ms = (perf_counter() - materialization_started) * 1000

        return AvsSearchResponse(
            results=results,
            latency=AvsSearchLatency(
                retrieval_ms=remote.retrieval_ms,
                coverage_ms=coverage_ms,
                materialization_ms=materialization_ms,
                total_ms=(perf_counter() - started) * 1000,
            ),
            candidate_pool_size=len(remote.candidates),
            deduplicated_candidate_count=len(ordered),
            unique_videos=len({item.video_id for item in results}),
            warnings=list(remote.warnings),
        )
