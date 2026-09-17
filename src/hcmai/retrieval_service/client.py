"""Synchronous gRPC client for the standalone retrieval service.

This module provides one long-lived client with explicit deadlines, health
probing, and domain encoding/decoding. It has no retry behavior.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.orchestration.workflows.temporal_search import (
    SelectedVideoScoreResult,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import KISRetrievalPlan
from hcmai.retrieval_service.codec import (
    decode_artifact,
    decode_decoder_config,
    decode_result,
    decode_video_scores,
    encode_plan,
)
from hcmai.retrieval_service.config import RetrievalClientSettings
from hcmai.retrieval_service.errors import map_rpc_error
from hcmai.retrieval_service.proto import retrieval_pb2, retrieval_pb2_grpc
from hcmai.retrieval_service.remote import RemoteImageCandidate, RemoteImageSearchResult

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RemoteRetrievalStatus:
    """Non-throwing health and capability summary for monitoring and health routes."""

    target: str
    reachable: bool
    ready: bool
    scoring_revision: str | None
    active_modalities: tuple[str, ...]
    startup_messages: tuple[str, ...]


class RetrievalGrpcClient:
    """Manage connection and unary RPC execution to RetrievalService."""

    def __init__(self, settings: RetrievalClientSettings) -> None:
        """Bind settings and construct one long-lived channel."""
        self.settings = settings
        options = (
            ("grpc.max_send_message_length", settings.max_message_bytes),
            ("grpc.max_receive_message_length", settings.max_message_bytes),
        )
        self.channel = grpc.insecure_channel(settings.target, options=options)
        self.stub = retrieval_pb2_grpc.RetrievalServiceStub(self.channel)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)

    def probe(self) -> RemoteRetrievalStatus:
        """Perform a non-throwing health and capability probe for diagnostics."""
        try:
            health_req = health_pb2.HealthCheckRequest(
                service="hcmai.retrieval.v1.RetrievalService"
            )
            health_resp = self.health_stub.Check(
                health_req,
                timeout=self.settings.health_timeout_seconds,
            )
            is_serving = health_resp.status == health_pb2.HealthCheckResponse.SERVING
            if not is_serving:
                try:
                    cap_resp = self.stub.GetCapabilities(
                        retrieval_pb2.GetCapabilitiesRequest(),
                        timeout=self.settings.health_timeout_seconds,
                    )
                    return RemoteRetrievalStatus(
                        target=self.settings.target,
                        reachable=True,
                        ready=False,
                        scoring_revision=cap_resp.scoring_revision or None,
                        active_modalities=tuple(cap_resp.active_modalities),
                        startup_messages=tuple(cap_resp.startup_messages),
                    )
                except grpc.RpcError:
                    return RemoteRetrievalStatus(
                        target=self.settings.target,
                        reachable=True,
                        ready=False,
                        scoring_revision=None,
                        active_modalities=(),
                        startup_messages=("Retrieval service is NOT_SERVING",),
                    )

            cap_resp = self.stub.GetCapabilities(
                retrieval_pb2.GetCapabilitiesRequest(),
                timeout=self.settings.health_timeout_seconds,
            )
            return RemoteRetrievalStatus(
                target=self.settings.target,
                reachable=True,
                ready=cap_resp.ready,
                scoring_revision=cap_resp.scoring_revision or None,
                active_modalities=tuple(cap_resp.active_modalities),
                startup_messages=tuple(cap_resp.startup_messages),
            )
        except grpc.RpcError as error:
            logger.debug(
                "Retrieval probe failed for %s: %s",
                self.settings.target,
                error,
            )
            return RemoteRetrievalStatus(
                target=self.settings.target,
                reachable=False,
                ready=False,
                scoring_revision=None,
                active_modalities=(),
                startup_messages=(
                    f"Failed to connect to retrieval service at {self.settings.target}",
                ),
            )

    def get_capabilities(self) -> RemoteRetrievalStatus:
        """Fetch remote retrieval capabilities or raise typed transport errors."""
        try:
            cap_resp = self.stub.GetCapabilities(
                retrieval_pb2.GetCapabilitiesRequest(),
                timeout=self.settings.health_timeout_seconds,
            )
            return RemoteRetrievalStatus(
                target=self.settings.target,
                reachable=True,
                ready=cap_resp.ready,
                scoring_revision=cap_resp.scoring_revision or None,
                active_modalities=tuple(cap_resp.active_modalities),
                startup_messages=tuple(cap_resp.startup_messages),
            )
        except grpc.RpcError as error:
            raise map_rpc_error(error) from error

    def search_plan(
        self,
        plan: KISRetrievalPlan,
        *,
        corpus: Corpus,
        use_dense: bool = True,
        use_bm25: bool = False,
        top_k: int = 20,
    ) -> TemporalSearchArtifact:
        """Call remote SearchPlan and decode canonical artifact."""
        request = encode_plan(
            plan,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )
        try:
            response = self.stub.SearchPlan(
                request,
                timeout=self.settings.timeout_seconds,
            )
            return decode_artifact(
                response,
                expected_event_count=plan.event_count,
                corpus=corpus,
            )
        except grpc.RpcError as error:
            raise map_rpc_error(error) from error

    def search_events(
        self,
        original_events: Sequence[str],
        *,
        corpus: Corpus,
        top_k: int = 20,
        retrieval_events: Sequence[str] | None = None,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> TemporalSearchResult:
        """Call remote SearchEvents (TRAKE) and decode result."""
        request = retrieval_pb2.SearchEventsRequest(
            original_events=list(original_events),
            retrieval_events=list(retrieval_events) if retrieval_events else [],
            caption_events=list(caption_events) if caption_events else [],
            has_caption_events=caption_events is not None,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=top_k,
        )
        try:
            response = self.stub.SearchEvents(
                request,
                timeout=self.settings.timeout_seconds,
            )
            return decode_result(
                response,
                corpus=corpus,
                expected_event_count=len(original_events),
            )
        except grpc.RpcError as error:
            raise map_rpc_error(error) from error

    def score_video(
        self,
        plan: KISRetrievalPlan,
        *,
        video_id: str,
        corpus: Corpus,
        use_dense: bool = True,
        use_bm25: bool = False,
    ) -> SelectedVideoScoreResult:
        """Call remote ScoreVideo and decode selected video scores."""
        plan_req = encode_plan(
            plan,
            use_dense=use_dense,
            use_bm25=use_bm25,
            top_k=20,
        )
        request = retrieval_pb2.ScoreVideoRequest(
            plan=plan_req,
            video_id=video_id,
        )
        try:
            response = self.stub.ScoreVideo(
                request,
                timeout=self.settings.timeout_seconds,
            )
            video = decode_video_scores(
                response.video_scores,
                expected_event_count=plan.event_count,
                corpus=corpus,
            )
            decoder_config = decode_decoder_config(response.decoder_config)
            return SelectedVideoScoreResult(
                video=video,
                retrieval_ms=response.retrieval_ms,
                decoder_config=decoder_config,
                scoring_revision=response.scoring_revision or None,
            )
        except grpc.RpcError as error:
            raise map_rpc_error(error) from error

    def search_image(
        self,
        payload: bytes,
        *,
        content_type: str | None,
        top_k: int,
    ) -> RemoteImageSearchResult:
        """Call remote SearchImage and return transport-neutral candidates."""
        request = retrieval_pb2.SearchImageRequest(
            payload=payload,
            content_type=content_type or "",
            top_k=top_k,
        )
        try:
            response = self.stub.SearchImage(
                request,
                timeout=self.settings.timeout_seconds,
            )
            candidates = tuple(
                RemoteImageCandidate(
                    video_id=c.video_id,
                    frame_id=c.frame_id,
                    frame_idx=c.frame_idx,
                    timestamp_ms=c.timestamp_ms,
                    score=c.score,
                )
                for c in response.candidates
            )
            return RemoteImageSearchResult(
                candidates=candidates,
                query_ms=response.query_ms,
                retrieval_ms=response.retrieval_ms,
            )
        except grpc.RpcError as error:
            raise map_rpc_error(error) from error

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self.channel.close()
