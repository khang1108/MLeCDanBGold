"""ASR and diarization routes for hosted inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, Request

from offline.enrichment.inference_contracts import (
    AudioReferenceRequest,
    DiarizationRequest,
    TranscriptInferenceResponse,
)
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable

router = APIRouter(prefix="/v1/transcripts", tags=["transcripts"])


@router.post("/asr", response_model=TranscriptInferenceResponse)
async def transcribe(
    payload: AudioReferenceRequest,
    request: Request,
) -> TranscriptInferenceResponse:
    """Transcribe a validated audio reference into timestamped segments."""

    started = perf_counter()
    runtime = runtime_from(request)
    try:
        segments = runtime.transcribe_reference(payload)
        model_status = loaded_model_status(runtime, "asr")
    except Exception as error:
        raise unavailable("ASR inference failed", error) from error
    return TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model=model_status.checkpoint or "asr",
        revision=model_status.revision,
        segments=segments,
        latency_ms=(perf_counter() - started) * 1_000,
    )


@router.post("/diarization", response_model=TranscriptInferenceResponse)
async def diarize(
    payload: DiarizationRequest,
    request: Request,
) -> TranscriptInferenceResponse:
    """Assign speakers to existing transcript segments for an audio reference."""

    started = perf_counter()
    runtime = runtime_from(request)
    try:
        segments = runtime.diarize_reference(payload)
        model_status = loaded_model_status(runtime, "diarization")
    except Exception as error:
        raise unavailable("Diarization inference failed", error) from error
    return TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model=model_status.checkpoint or "diarization",
        revision=model_status.revision,
        segments=segments,
        latency_ms=(perf_counter() - started) * 1_000,
    )
