"""HTTP client for communicating with the standalone retrieval service.

This module replaces the gRPC client with a lightweight httpx-based client
to query the retrieval process running on 127.0.0.1:8002.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import httpx

from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval.serving.schemas import (
    CapabilitiesResponse,
    ImageSearchCandidatesSchema,
    ScoreVideoRequestSchema,
    SearchEventsRequestSchema,
    SelectedVideoScoreSchema,
    TemporalSearchArtifactSchema,
    TemporalSearchResultSchema,
    TextCandidateSchema,
    TextSearchCandidatesSchema,
    TextSearchRequestSchema,
)
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.utils.errors import (
    RetrievalClientError,
    RetrievalUnavailableError,
    map_http_error,
)
from hcmai.retrieval.serving.utils.serialization import (
    plan_to_schema,
    schema_to_artifact,
    schema_to_path,
    schema_to_video_scores,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RemoteRetrievalStatus:
    """Connection diagnostics and capabilities of a remote retrieval target."""

    target: str
    reachable: bool
    ready: bool
    scoring_revision: str | None
    active_modalities: tuple[str, ...]
    startup_messages: tuple[str, ...]


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


@dataclass(frozen=True, slots=True)
class RemoteTextCandidate:
    frame_id: str
    rank: int
    score: float | None


@dataclass(frozen=True, slots=True)
class RemoteTextSearchResult:
    candidates: tuple[RemoteTextCandidate, ...]
    retrieval_ms: float
    warnings: tuple[str, ...]


class RetrievalHttpClient:
    """Manage HTTP requests and response mapping to the retrieval service."""

    def __init__(self, settings: RetrievalClientSettings) -> None:
        """Bind settings and construct an httpx client."""
        self.settings = settings
        self.base_url = settings.base_url
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=settings.timeout_seconds,
        )

    def close(self) -> None:
        """Close underlying connection pool."""
        self._client.close()

    def __enter__(self) -> RetrievalHttpClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def probe(self) -> RemoteRetrievalStatus:
        """Perform a non-throwing health and capability probe for diagnostics."""
        try:
            resp = self._client.get(
                "/capabilities",
                timeout=self.settings.health_timeout_seconds,
            )
            if resp.status_code == 200:
                data = CapabilitiesResponse.model_validate(resp.json())
                return RemoteRetrievalStatus(
                    target=self.settings.target,
                    reachable=True,
                    ready=data.ready,
                    scoring_revision=data.scoring_revision,
                    active_modalities=tuple(data.active_modalities),
                    startup_messages=tuple(data.startup_messages),
                )
            return RemoteRetrievalStatus(
                target=self.settings.target,
                reachable=True,
                ready=False,
                scoring_revision=None,
                active_modalities=(),
                startup_messages=(f"Status {resp.status_code}: {resp.text}",),
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as err:
            logger.debug("Retrieval probe failed for %s: %s", self.base_url, err)
            return RemoteRetrievalStatus(
                target=self.settings.target,
                reachable=False,
                ready=False,
                scoring_revision=None,
                active_modalities=(),
                startup_messages=(str(err),),
            )

    def get_capabilities(self) -> CapabilitiesResponse:
        """Fetch declared capabilities from the remote retrieval service."""
        try:
            resp = self._client.get("/capabilities")
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            return CapabilitiesResponse.model_validate(resp.json())
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to connect to retrieval service at {self.base_url}: {err}"
            ) from err

    def search_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        corpus: Corpus | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        """Execute multimodal KIS retrieval plan over HTTP."""
        req_schema = plan_to_schema(
            plan,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )
        try:
            resp = self._client.post(
                "/search_plan",
                json=req_schema.model_dump(),
            )
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            artifact_schema = TemporalSearchArtifactSchema.model_validate(resp.json())
            return schema_to_artifact(artifact_schema, corpus=corpus)
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to execute search_plan at {self.base_url}: {err}"
            ) from err

    def search_events(
        self,
        original_events: Sequence[str],
        *,
        corpus: Corpus | None = None,
        top_k: int,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        """Execute ordered event queries (TRAKE) over HTTP."""
        req_schema = SearchEventsRequestSchema(
            original_events=list(original_events),
            retrieval_events=list(retrieval_events) if retrieval_events is not None else None,
            caption_events=list(caption_events) if caption_events is not None else None,
            has_caption_events=caption_events is not None,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )
        try:
            resp = self._client.post(
                "/search_events",
                json=req_schema.model_dump(),
            )
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            result_schema = TemporalSearchResultSchema.model_validate(resp.json())
            paths = tuple(schema_to_path(p) for p in result_schema.paths)
            return TemporalSearchResult(
                paths=paths,
                retrieval_ms=result_schema.retrieval_ms,
                alignment_ms=result_schema.alignment_ms,
            )
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to execute search_events at {self.base_url}: {err}"
            ) from err

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        corpus: Corpus | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        """Score one selected video over HTTP."""
        plan_schema = plan_to_schema(
            plan,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=20,
        )
        req_schema = ScoreVideoRequestSchema(
            plan=plan_schema,
            video_id=video_id,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        try:
            resp = self._client.post(
                "/score_video",
                json=req_schema.model_dump(),
            )
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            data = SelectedVideoScoreSchema.model_validate(resp.json())
            video_scores = schema_to_video_scores(data.video, corpus=corpus)
            decoder_config = DecoderConfigSnapshot(
                lambda_gap=data.decoder_config.lambda_gap,
                event_power=data.decoder_config.event_power,
                cluster_delta=data.decoder_config.cluster_delta,
                path_min_separation_ms=data.decoder_config.path_min_separation_ms,
            )
            return SelectedVideoScoreResult(
                video=video_scores,
                retrieval_ms=data.retrieval_ms,
                decoder_config=decoder_config,
                scoring_revision=data.scoring_revision,
            )
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to score_video at {self.base_url}: {err}"
            ) from err

    def search_image(
        self,
        payload: bytes,
        *,
        content_type: str | None = None,
        top_k: int = 20,
    ) -> RemoteImageSearchResult:
        """Execute direct visual query search over HTTP."""
        headers = {"content-type": content_type or "image/jpeg"}
        try:
            resp = self._client.post(
                "/search_image",
                params={"top_k": top_k},
                content=payload,
                headers=headers,
            )
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            data = ImageSearchCandidatesSchema.model_validate(resp.json())
            candidates = tuple(
                RemoteImageCandidate(
                    video_id=c.video_id,
                    frame_id=c.frame_id,
                    frame_idx=c.frame_idx,
                    timestamp_ms=c.timestamp_ms,
                    score=c.score,
                )
                for c in data.candidates
            )
            return RemoteImageSearchResult(
                candidates=candidates,
                query_ms=data.query_ms,
                retrieval_ms=data.retrieval_ms,
            )
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to search_image at {self.base_url}: {err}"
            ) from err

    def search_text(self, query: str, *, top_k: int = 100) -> RemoteTextSearchResult:
        req = TextSearchRequestSchema(query=query, top_k=top_k)
        try:
            resp = self._client.post("/search_text", json=req.model_dump())
            if resp.status_code != 200:
                raise map_http_error(resp.status_code, resp.text)
            data = TextSearchCandidatesSchema.model_validate(resp.json())
            return RemoteTextSearchResult(
                candidates=tuple(
                    RemoteTextCandidate(frame_id=item.frame_id, rank=item.rank, score=item.score)
                    for item in data.candidates
                ),
                retrieval_ms=data.retrieval_ms,
                warnings=tuple(data.warnings),
            )
        except httpx.RequestError as err:
            raise RetrievalUnavailableError(
                f"Failed to search_text at {self.base_url}: {err}"
            ) from err
