"""ASR and diarization routes for hosted inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from time import perf_counter

from fastapi import APIRouter, File, Form, Request, UploadFile

from llm.contracts import (
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


@router.post("/asr-file", response_model=TranscriptInferenceResponse)
async def transcribe_file(
    request: Request,
    request_id: str = Form(...),
    video_id: str = Form(...),
    sample_rate: int = Form(16_000),
    audio_sha256: str = Form(...),
    audio: UploadFile = File(...),
) -> TranscriptInferenceResponse:
    """Transcribe an uploaded local FLAC without any S3/public URL handoff."""

    started = perf_counter()
    if not request_id.strip() or not video_id.strip():
        raise unavailable("ASR upload is invalid", ValueError("blank request/video identity"))
    if sample_rate < 8_000 or sample_rate > 48_000:
        raise unavailable("ASR upload is invalid", ValueError("unsupported sample rate"))
    if len(audio_sha256) != 64:
        raise unavailable("ASR upload is invalid", ValueError("invalid audio checksum"))

    maximum = int(os.getenv("HCMAI_MAX_AUDIO_BYTES", str(1024 * 1024 * 1024)))
    runtime = runtime_from(request)
    try:
        with tempfile.TemporaryDirectory(prefix="hcmai-upload-") as directory:
            suffix = Path(audio.filename or "audio.flac").suffix or ".flac"
            target = Path(directory) / f"audio{suffix}"
            digest = hashlib.sha256()
            total = 0
            with target.open("wb") as handle:
                while True:
                    chunk = await audio.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > maximum:
                        raise ValueError("uploaded audio exceeds configured byte limit")
                    digest.update(chunk)
                    handle.write(chunk)
            if total == 0:
                raise ValueError("uploaded audio is empty")
            if digest.hexdigest() != audio_sha256:
                raise ValueError("uploaded audio checksum mismatch")
            segments = runtime.transcribe_file(target, video_id)
            model_status = loaded_model_status(runtime, "asr")
    except Exception as error:
        raise unavailable("ASR inference failed", error) from error
    finally:
        await audio.close()

    return TranscriptInferenceResponse(
        request_id=request_id,
        video_id=video_id,
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
