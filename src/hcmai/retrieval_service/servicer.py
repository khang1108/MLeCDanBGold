"""gRPC servicer for hcmai.retrieval.v1.RetrievalService.

This module validates wire requests, delegates execution to the heavy
RetrievalRuntime facade, maps domain errors to standard gRPC status codes,
and encodes domain outputs through codec.py. It contains no retrieval logic.
"""

from __future__ import annotations

from collections.abc import Sequence
import grpc

from hcmai.common.utils.logging import get_logger
from hcmai.orchestration.workflows.image_search import ImageQueryTooLargeError
from hcmai.retrieval_service.codec import (
    decode_plan,
    encode_decoder_config,
    encode_temporal_search_artifact,
    encode_temporal_search_result,
    encode_video_scores,
)
from hcmai.retrieval_service.proto import retrieval_pb2, retrieval_pb2_grpc
from hcmai.retrieval_service.runtime import RetrievalRuntime

logger = get_logger(__name__)


class RetrievalServicer(retrieval_pb2_grpc.RetrievalServiceServicer):
    """Serve private retrieval operations over gRPC."""

    def __init__(
        self,
        runtime: RetrievalRuntime | None,
        *,
        startup_messages: Sequence[str] = (),
    ) -> None:
        """Bind the runtime facade and optional diagnostic messages."""
        self.runtime = runtime
        self.startup_messages = tuple(startup_messages)

    def _require_runtime(self, context: grpc.ServicerContext) -> RetrievalRuntime:
        """Abort with UNAVAILABLE if the retrieval runtime is not loaded."""
        if self.runtime is None:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "Retrieval runtime is not available",
            )
        return self.runtime

    def GetCapabilities(
        self,
        request: retrieval_pb2.GetCapabilitiesRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_pb2.GetCapabilitiesResponse:
        """Return diagnostic capabilities and limits."""
        try:
            if self.runtime is None:
                return retrieval_pb2.GetCapabilitiesResponse(
                    ready=False,
                    scoring_revision="",
                    active_modalities=[],
                    startup_messages=list(self.startup_messages),
                    max_temporal_event_count=0,
                    image_max_upload_bytes=0,
                    image_max_pixels=0,
                )
            caps = self.runtime.capabilities()
            return retrieval_pb2.GetCapabilitiesResponse(
                ready=caps.ready,
                scoring_revision=caps.scoring_revision,
                active_modalities=list(caps.active_modalities),
                startup_messages=list(caps.startup_messages),
                max_temporal_event_count=caps.max_temporal_event_count,
                image_max_upload_bytes=caps.image_max_upload_bytes,
                image_max_pixels=caps.image_max_pixels,
            )
        except Exception as error:
            logger.exception("GetCapabilities failed: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "internal retrieval error")

    def SearchPlan(
        self,
        request: retrieval_pb2.SearchPlanRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_pb2.TemporalSearchArtifact:
        """Execute multimodal KIS retrieval plan and return bounded score artifact."""
        runtime = self._require_runtime(context)
        try:
            if len(request.events) == 0:
                raise ValueError("SearchPlanRequest must contain at least one event")
            plan = decode_plan(request)
            artifact = runtime.search_plan(
                plan,
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
                top_k=request.top_k or 20,
            )
            return encode_temporal_search_artifact(artifact)
        except ImageQueryTooLargeError as error:
            logger.warning("SearchPlan input too large: %s", error)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except ValueError as error:
            logger.warning("SearchPlan invalid argument: %s", error)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except KeyError as error:
            logger.warning("SearchPlan resource not found: %s", error)
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except Exception as error:
            logger.exception("SearchPlan unexpected error: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "internal retrieval error")

    def SearchEvents(
        self,
        request: retrieval_pb2.SearchEventsRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_pb2.TemporalSearchResult:
        """Execute ordered event temporal search (TRAKE)."""
        runtime = self._require_runtime(context)
        try:
            if len(request.original_events) == 0:
                raise ValueError("SearchEventsRequest must contain at least one event")
            result = runtime.search_events(
                request.original_events,
                top_k=request.top_k or 20,
                retrieval_events=list(request.retrieval_events) or None,
                caption_events=list(request.caption_events) if request.has_caption_events else None,
                use_dense=request.use_dense,
                use_bm25=request.use_bm25,
            )
            return encode_temporal_search_result(
                result,
                scoring_revision=runtime.scoring_revision,
            )
        except ValueError as error:
            logger.warning("SearchEvents invalid argument: %s", error)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except KeyError as error:
            logger.warning("SearchEvents resource not found: %s", error)
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except Exception as error:
            logger.exception("SearchEvents unexpected error: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "internal retrieval error")

    def ScoreVideo(
        self,
        request: retrieval_pb2.ScoreVideoRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_pb2.SelectedVideoScore:
        """Score one selected video under the plan and return its snapshot."""
        runtime = self._require_runtime(context)
        try:
            if not request.video_id:
                raise ValueError("ScoreVideoRequest video_id must not be empty")
            if len(request.plan.events) == 0:
                raise ValueError("ScoreVideoRequest plan must contain at least one event")
            plan = decode_plan(request.plan)
            selected = runtime.score_video(
                plan,
                video_id=request.video_id,
                use_dense=request.plan.use_dense,
                use_bm25=request.plan.use_bm25,
            )
            return retrieval_pb2.SelectedVideoScore(
                video_scores=encode_video_scores(selected.video),
                retrieval_ms=selected.retrieval_ms,
                decoder_config=encode_decoder_config(selected.decoder_config),
                scoring_revision=selected.scoring_revision or "",
            )
        except ImageQueryTooLargeError as error:
            logger.warning("ScoreVideo input too large: %s", error)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except ValueError as error:
            logger.warning("ScoreVideo invalid argument: %s", error)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except KeyError as error:
            logger.warning("ScoreVideo video not found: %s", error)
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except Exception as error:
            logger.exception("ScoreVideo unexpected error: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "internal retrieval error")

    def SearchImage(
        self,
        request: retrieval_pb2.SearchImageRequest,
        context: grpc.ServicerContext,
    ) -> retrieval_pb2.ImageSearchCandidates:
        """Execute direct image query search."""
        runtime = self._require_runtime(context)
        try:
            if not request.payload:
                raise ValueError("SearchImageRequest payload must not be empty")
            if request.top_k <= 0:
                raise ValueError("SearchImageRequest top_k must be greater than zero")
            response = runtime.search_image(
                request.payload,
                content_type=request.content_type or None,
                top_k=request.top_k,
            )
            candidates = [
                retrieval_pb2.ImageCandidate(
                    video_id=result.video_id,
                    frame_id=result.frame_id,
                    frame_idx=result.frame_idx,
                    timestamp_ms=result.timestamp_ms,
                    score=result.score,
                )
                for result in response.results
            ]
            return retrieval_pb2.ImageSearchCandidates(
                candidates=candidates,
                query_ms=response.latency.query_ms,
                retrieval_ms=response.latency.retrieval_ms,
            )
        except ImageQueryTooLargeError as error:
            logger.warning("SearchImage payload too large: %s", error)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except ValueError as error:
            logger.warning("SearchImage invalid argument: %s", error)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except KeyError as error:
            logger.warning("SearchImage resource not found: %s", error)
            context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except Exception as error:
            logger.exception("SearchImage unexpected error: %s", type(error).__name__)
            context.abort(grpc.StatusCode.INTERNAL, "internal retrieval error")
