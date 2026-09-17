"""Transport-neutral remote adapters for temporal and image search.

This module owns client-side adaptation of gRPC retrieval stubs into the
TemporalSearchGateway protocol and ImageSearchService API surface. It validates
media bounds locally, materializes canonical results via Corpus, and preserves
canonical identity invariants.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING
import numpy as np

from hcmai.api.contracts import ImageSearchResponse, SearchLatency, SearchResult
from hcmai.common.config import AlignmentConfig
from hcmai.corpus import Corpus
from hcmai.orchestration.utils.materializer import SearchMaterializer
from hcmai.orchestration.workflows.image_search import (
    ImageQueryTooLargeError,
    InvalidImageQueryError,
)
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
    decode_video_scores,
)
from hcmai.retrieval.evidence.components import TemporalScoreComponent
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval_service.errors import (
    RetrievalInvalidRequestError,
    RetrievalTooLargeError,
    RetrievalUnavailableError,
)
from hcmai.temporal.dp import AlignedPath

if TYPE_CHECKING:
    from hcmai.retrieval_service.client import RetrievalGrpcClient


@dataclass(frozen=True, slots=True)
class RemoteImageCandidate:
    """Canonical frame candidate returned from remote visual retrieval."""

    video_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    score: float


@dataclass(frozen=True, slots=True)
class RemoteImageSearchResult:
    """Remote image retrieval result candidates and server timings."""

    candidates: tuple[RemoteImageCandidate, ...]
    query_ms: float
    retrieval_ms: float


class RemoteTemporalSearchService:
    """Delegate temporal retrieval and scoring to a remote gRPC service."""

    def __init__(self, corpus: Corpus, client: RetrievalGrpcClient) -> None:
        """Bind canonical corpus and remote gRPC client."""
        self.corpus = corpus
        self.client = client

    def search(
        self,
        original_events: Sequence[str],
        *,
        top_k: int,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        """Search ordered event queries over remote gRPC service."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        return self.client.search_events(
            original_events,
            corpus=self.corpus,
            top_k=top_k,
            retrieval_events=retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )

    def search_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchResult:
        """Search multimodal plan and return canonical aligned paths."""
        return self.search_plan_artifact(
            plan,
            image_component=image_component,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        ).result

    def search_plan_artifact(
        self,
        plan: KISRetrievalPlan,
        *,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        """Search multimodal plan and return paths plus exact score matrix."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if image_component is not None:
            raise NotImplementedError(
                "Remote temporal search does not accept local image_component"
            )
        return self.client.search_plan(
            plan,
            corpus=self.corpus,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        image_component: TemporalScoreComponent | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        """Score one selected video under plan without returning all corpus scores."""
        if not video_id:
            raise ValueError("video_id must not be empty")
        if image_component is not None:
            raise NotImplementedError(
                "Remote temporal search does not accept local image_component"
            )
        return self.client.score_video(
            plan,
            video_id=video_id,
            corpus=self.corpus,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )

    def decode_video(
        self,
        video: VideoEventScores,
        *,
        allowed: np.ndarray,
        decoder_config: DecoderConfigSnapshot | None = None,
    ) -> tuple[AlignedPath, ...]:
        """Decode one scored video locally without making network calls."""
        if decoder_config is None:
            default_config = AlignmentConfig()
            decoder_config = DecoderConfigSnapshot(
                lambda_gap=default_config.lambda_gap,
                event_power=default_config.event_power,
                cluster_delta=default_config.cluster_delta,
                path_min_separation_ms=default_config.path_min_separation_ms,
            )
        return decode_video_scores(
            self.corpus,
            video,
            allowed=allowed,
            decoder_config=decoder_config,
        )

    def get_scoring_revision(self) -> str:
        """Fetch active remote scoring revision or raise if unready."""
        status = self.client.get_capabilities()
        if not status.ready or not status.scoring_revision:
            raise RetrievalUnavailableError(
                "Remote retrieval service is not ready or has no scoring revision"
            )
        return status.scoring_revision


class RemoteImageSearchService:
    """Delegate visual query retrieval to remote standalone service."""

    SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

    def __init__(
        self,
        corpus: Corpus,
        client: RetrievalGrpcClient,
        *,
        max_upload_bytes: int,
        max_pixels: int,
    ) -> None:
        """Bind canonical corpus, gRPC client, and upload bounds."""
        if max_upload_bytes <= 0 or max_pixels <= 0:
            raise ValueError("image upload limits must be positive")
        self.corpus = corpus
        self.client = client
        self.max_upload_bytes = max_upload_bytes
        self.max_pixels = max_pixels
        self.materializer = SearchMaterializer(corpus)

    def search(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> ImageSearchResponse:
        """Validate input bounds, execute remote search, and materialize results."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if content_type not in self.SUPPORTED_MEDIA_TYPES:
            raise InvalidImageQueryError(
                "image must use JPEG, PNG, or WebP media type"
            )
        if not payload:
            raise InvalidImageQueryError("image payload must not be empty")
        if len(payload) > self.max_upload_bytes:
            raise ImageQueryTooLargeError(
                f"image payload exceeds {self.max_upload_bytes} bytes"
            )

        started = perf_counter()
        try:
            remote_result = self.client.search_image(
                payload,
                content_type=content_type,
                top_k=top_k,
            )
        except RetrievalTooLargeError as error:
            raise ImageQueryTooLargeError(str(error)) from error
        except RetrievalInvalidRequestError as error:
            raise InvalidImageQueryError(str(error)) from error

        materialization_started = perf_counter()
        results: list[SearchResult] = []
        for candidate in remote_result.candidates:
            frame = self.corpus.frame(candidate.frame_id)
            if frame.video_id != candidate.video_id:
                raise ValueError(
                    f"remote candidate video_id {candidate.video_id!r} conflicts with corpus {frame.video_id!r}"
                )
            if frame.frame_idx != candidate.frame_idx:
                raise ValueError(
                    f"remote candidate frame_idx {candidate.frame_idx} conflicts with corpus {frame.frame_idx}"
                )
            if frame.timestamp_ms != candidate.timestamp_ms:
                raise ValueError(
                    f"remote candidate timestamp_ms {candidate.timestamp_ms} conflicts with corpus {frame.timestamp_ms}"
                )

            path = AlignedPath(
                video_id=frame.video_id,
                score=candidate.score,
                frame_ids=(frame.frame_id,),
                frame_idxs=(frame.frame_idx,),
                timestamps_ms=(frame.timestamp_ms,),
            )
            results.append(self.materializer.build_kis_result(path))

        materialization_ms = (perf_counter() - materialization_started) * 1_000
        total_ms = (perf_counter() - started) * 1_000

        return ImageSearchResponse(
            results=results,
            latency=SearchLatency(
                query_ms=remote_result.query_ms,
                retrieval_ms=remote_result.retrieval_ms,
                materialization_ms=materialization_ms,
                total_ms=total_ms,
            ),
        )
