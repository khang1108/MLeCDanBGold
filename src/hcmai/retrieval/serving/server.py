"""HTTP server application and CLI entry point for retrieval serving.

This module exposes the standalone retrieval service over HTTP (FastAPI) on
127.0.0.1:8002, hosting RetrievalRuntime without Uvicorn reloads.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import sys
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
import uvicorn

from hcmai.common.environment import load_repository_environment
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retrieval.serving.runtime import RetrievalRuntime
from hcmai.retrieval.serving.schemas import (
    CapabilitiesResponse,
    DecoderConfigSchema,
    ImageCandidateSchema,
    ImageSearchCandidatesSchema,
    ScoreVideoRequestSchema,
    SearchEventsRequestSchema,
    SearchPlanRequestSchema,
    SelectedVideoScoreSchema,
    TemporalSearchArtifactSchema,
    TemporalSearchResultSchema,
)
from hcmai.retrieval.serving.utils.serialization import (
    artifact_to_schema,
    path_to_schema,
    schema_to_plan,
    video_scores_to_schema,
)

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8002


def create_app(
    runtime: RetrievalRuntime | None = None,
    *,
    startup_messages: Sequence[str] = (),
) -> FastAPI:
    """Create and configure one FastAPI retrieval server application."""
    app = FastAPI(title="HCMAI Retrieval Service", version="1.0.0")

    app.state.runtime = runtime
    app.state.startup_messages = list(startup_messages)

    def _get_runtime() -> RetrievalRuntime:
        rt = app.state.runtime
        if rt is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Retrieval service runtime is not ready",
            )
        return rt

    @app.get("/health")
    def health_check() -> JSONResponse:
        """Standard liveness probe."""
        if app.state.runtime is not None:
            return JSONResponse({"status": "SERVING"})
        return JSONResponse(
            {"status": "NOT_SERVING"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.get("/capabilities", response_model=CapabilitiesResponse)
    def get_capabilities() -> CapabilitiesResponse:
        """Diagnostic capability and limits advertised by the runtime."""
        rt = app.state.runtime
        if rt is None:
            return CapabilitiesResponse(
                ready=False,
                scoring_revision=None,
                active_modalities=[],
                startup_messages=list(app.state.startup_messages),
            )
        caps = rt.capabilities()
        return CapabilitiesResponse(
            ready=caps.ready,
            scoring_revision=caps.scoring_revision,
            active_modalities=list(caps.active_modalities),
            startup_messages=list(caps.startup_messages),
            max_temporal_event_count=caps.max_temporal_event_count,
            image_max_upload_bytes=caps.image_max_upload_bytes,
            image_max_pixels=caps.image_max_pixels,
        )

    @app.post("/search_plan", response_model=TemporalSearchArtifactSchema)
    def search_plan(req: SearchPlanRequestSchema) -> TemporalSearchArtifactSchema:
        """Execute multimodal KIS retrieval plan and return temporal paths and scores."""
        rt = _get_runtime()
        try:
            plan = schema_to_plan(req)
            artifact = rt.search_plan(
                plan,
                use_dense=req.use_dense,
                use_bm25=req.use_bm25,
                top_k=req.top_k,
            )
            return artifact_to_schema(artifact)
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        except Exception as err:
            logger.exception("Error in /search_plan: %s", err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(err),
            ) from err

    @app.post("/search_events", response_model=TemporalSearchResultSchema)
    def search_events(req: SearchEventsRequestSchema) -> TemporalSearchResultSchema:
        """Execute ordered event temporal search (TRAKE)."""
        rt = _get_runtime()
        try:
            result = rt.search_events(
                req.original_events,
                top_k=req.top_k,
                retrieval_events=req.retrieval_events,
                caption_events=req.caption_events,
                use_dense=req.use_dense,
                use_bm25=req.use_bm25,
            )
            return TemporalSearchResultSchema(
                paths=[path_to_schema(p) for p in result.paths],
                retrieval_ms=result.retrieval_ms,
                alignment_ms=result.alignment_ms,
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        except Exception as err:
            logger.exception("Error in /search_events: %s", err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(err),
            ) from err

    @app.post("/score_video", response_model=SelectedVideoScoreSchema)
    def score_video(req: ScoreVideoRequestSchema) -> SelectedVideoScoreSchema:
        """Score one targeted video under a retrieval plan."""
        rt = _get_runtime()
        try:
            plan = schema_to_plan(req.plan)
            selected = rt.score_video(
                plan,
                video_id=req.video_id,
                use_dense=req.use_dense,
                use_bm25=req.use_bm25,
            )
            if selected is None or selected.video is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Video {req.video_id!r} not found or could not be scored",
                )
            return SelectedVideoScoreSchema(
                video=video_scores_to_schema(selected.video),
                retrieval_ms=selected.retrieval_ms,
                decoder_config=DecoderConfigSchema(
                    lambda_gap=selected.decoder_config.lambda_gap,
                    event_power=selected.decoder_config.event_power,
                    cluster_delta=selected.decoder_config.cluster_delta,
                    path_min_separation_ms=selected.decoder_config.path_min_separation_ms,
                ),
                scoring_revision=selected.scoring_revision,
            )
        except HTTPException:
            raise
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        except Exception as err:
            logger.exception("Error in /score_video: %s", err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(err),
            ) from err

    @app.post("/search_image", response_model=ImageSearchCandidatesSchema)
    async def search_image(
        request: Request,
        top_k: int = Query(default=20, ge=1),
    ) -> ImageSearchCandidatesSchema:
        """Execute direct visual query retrieval."""
        rt = _get_runtime()
        content_type = request.headers.get("content-type", "").split(";")[0].strip()

        if content_type == "multipart/form-data":
            form = await request.form()
            file_item = form.get("image")
            if not isinstance(file_item, UploadFile):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing 'image' upload file in multipart request",
                )
            payload = await file_item.read()
            media_type = file_item.content_type or "image/jpeg"
        else:
            payload = await request.body()
            media_type = content_type or "image/jpeg"

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image payload must not be empty",
            )

        if len(payload) > rt.image_max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image payload exceeds {rt.image_max_upload_bytes} bytes",
            )

        try:
            resp = rt.search_image(
                payload,
                content_type=media_type,
                top_k=top_k,
            )
            candidates = [
                ImageCandidateSchema(
                    video_id=r.video_id,
                    frame_id=r.frame_id,
                    frame_idx=r.frame_idx,
                    timestamp_ms=r.timestamp_ms,
                    score=r.score,
                )
                for r in resp.results
            ]
            return ImageSearchCandidatesSchema(
                candidates=candidates,
                query_ms=resp.latency.query_ms,
                retrieval_ms=resp.latency.retrieval_ms,
            )
        except ValueError as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        except Exception as err:
            logger.exception("Error in /search_image: %s", err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(err),
            ) from err

    return app


def main() -> None:
    """CLI entry point for standalone retrieval HTTP service."""
    load_repository_environment()
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="HCMAI Standalone Retrieval HTTP Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    startup_messages: list[str] = []
    logger.info("Loading heavy retrieval runtime...")
    runtime = RetrievalRuntime.load(startup_messages)

    if runtime is None:
        logger.error("Failed to load retrieval runtime: %s", startup_messages)
        app = create_app(None, startup_messages=startup_messages)
    else:
        logger.info("Retrieval runtime ready (revision=%s)", runtime.scoring_revision)
        app = create_app(runtime, startup_messages=startup_messages)

    logger.info("Starting standalone HTTP retrieval server on %s:%d (single-worker, no reload)", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
