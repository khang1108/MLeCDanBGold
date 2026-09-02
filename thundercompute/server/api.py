"""Private GPU inference API served through the configured tunnel."""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import numpy as np
from PIL import Image

from hcmai.retrieval.embedding.inference_contracts import (
    EmbeddingResponse,
    TextEmbeddingResponse,
)
from offline.enrichment.inference_contracts import (
    AudioReferenceRequest,
    CaptionItem,
    CaptionResponse,
    DiarizationRequest,
    InferenceReadiness,
    OCRItem,
    OCRRegionItem,
    OCRResponse,
    TranscriptInferenceResponse,
)
from offline.enrichment.ocr.models.entities import json_safe_ocr_raw
from thundercompute.pipeline import LLMService
from thundercompute.contracts import (
    BoundaryScoreResponse,
    QueryCandidatesRequest,
    QueryCandidatesResponse,
    QueryEventsRequest,
    QueryTranslationResponse,
    RerankItem,
    RerankResponse,
    TextEmbeddingRequest,
)


def create_llm_app(runtime: LLMService | None = None) -> FastAPI:
    """Create an injectable inference API; production models load in lifespan."""
    owned = runtime or LLMService.from_environment()
    app = FastAPI(
        title="HCMAI Private Model Inference",
        version="0.1.0",
        lifespan=_lifespan(owned),
    )
    app.state.runtime = owned
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route(
        "/v1/captions",
        caption,
        methods=["POST"],
        response_model=CaptionResponse,
    )
    app.add_api_route(
        "/v1/enrichment/ocr",
        ocr,
        methods=["POST"],
        response_model=OCRResponse,
    )
    app.add_api_route(
        "/ready", ready, methods=["GET"], response_model=InferenceReadiness
    )
    app.add_api_route(
        "/v1/embeddings/text",
        embed,
        methods=["POST"],
        response_model=TextEmbeddingResponse,
    )
    app.add_api_route(
        "/v1/embeddings/images",
        embed_images,
        methods=["POST"],
        response_model=EmbeddingResponse,
    )
    app.add_api_route(
        "/v1/embeddings/dino",
        embed_dino,
        methods=["POST"],
        response_model=EmbeddingResponse,
    )
    app.add_api_route(
        "/v1/preprocessing/shot-scores",
        shot_scores,
        methods=["POST"],
        response_model=BoundaryScoreResponse,
    )
    app.add_api_route(
        "/v1/preprocessing/event-scores",
        event_scores,
        methods=["POST"],
        response_model=BoundaryScoreResponse,
        include_in_schema=False,
    )
    app.add_api_route(
        "/v1/preprocessing/event-window-scores",
        event_scores,
        methods=["POST"],
        response_model=BoundaryScoreResponse,
    )
    app.add_api_route(
        "/v1/transcripts/asr",
        transcribe,
        methods=["POST"],
        response_model=TranscriptInferenceResponse,
    )
    app.add_api_route(
        "/v1/transcripts/diarization",
        diarize,
        methods=["POST"],
        response_model=TranscriptInferenceResponse,
    )
    app.add_api_route(
        "/v1/rerank", rerank, methods=["POST"], response_model=RerankResponse
    )
    app.add_api_route(
        "/query-preparation/translate",
        translate_query_events,
        methods=["POST"],
        response_model=QueryTranslationResponse,
    )
    app.add_api_route(
        "/query-preparation/candidates",
        generate_query_candidates,
        methods=["POST"],
        response_model=QueryCandidatesResponse,
    )
    return app


def _lifespan(runtime: LLMService):
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runtime.load()
        yield
    return lifespan


async def health() -> dict[str, str]:
    return {"status": "ok"}


async def ready(request: Request) -> InferenceReadiness:
    value = request.app.state.runtime.readiness()
    if not value.ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return value


async def translate_query_events(
    payload: QueryEventsRequest, request: Request
) -> QueryTranslationResponse:
    """Translate ordered query events and reject provider shape drift."""

    try:
        events = request.app.state.runtime.translate_query_events(list(payload.events))
        response = QueryTranslationResponse(events=events)
        if len(response.events) != len(payload.events):
            raise ValueError("translation changed event count")
        return response
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Query translation failed: {(str(error) or type(error).__name__)[:160]}",
        ) from error


async def generate_query_candidates(
    payload: QueryCandidatesRequest, request: Request
) -> QueryCandidatesResponse:
    """Generate exactly five candidates aligned to the request events."""

    try:
        value = request.app.state.runtime.generate_query_candidates(
            list(payload.events), payload.candidate_count
        )
        response = QueryCandidatesResponse.model_validate(value)
        expected = len(payload.events)
        if len(response.literal_en) != expected or any(
            len(candidate) != expected for candidate in response.candidates
        ):
            raise ValueError("candidate generation changed event count")
        return response
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Query candidate generation failed: {(str(error) or type(error).__name__)[:160]}",
        ) from error


async def embed(
    payload: TextEmbeddingRequest, request: Request
) -> TextEmbeddingResponse:
    started = perf_counter()
    runtime = request.app.state.runtime
    embedding_config = (
        runtime.config.caption_embedding
        if payload.source == "text"
        else runtime.config.visual_embedding
    )
    if len(payload.texts) > embedding_config.batch_size:
        raise HTTPException(
            status_code=422,
            detail=(
                "text batch must contain 1.."
                f"{embedding_config.batch_size} items for source "
                f"{payload.source!r}"
            ),
        )
    try:
        vectors = runtime.embed_text(list(payload.texts), payload.source)
    except Exception as error:
        raise _unavailable("Embedding inference failed", error) from error
    return TextEmbeddingResponse(
        model=embedding_config.model_name,
        revision=embedding_config.revision,
        dimension=int(vectors.shape[1]),
        normalized=True,
        embeddings=vectors.tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def embed_images(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> EmbeddingResponse:
    """Endpoint tạo visual embedding (mặc định) cho một danh sách hình ảnh."""
    return await _embed_image_batch(request, item_ids, images, source="visual")


async def embed_dino(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> EmbeddingResponse:
    """Endpoint tạo DINO embedding cho một danh sách hình ảnh."""
    return await _embed_image_batch(request, item_ids, images, source="dino")


async def _embed_image_batch(
    request: Request,
    item_ids: str,
    images: list[UploadFile],
    *,
    source: str,
) -> EmbeddingResponse:
    """Logic xử lý chung để tính toán embedding từ hình ảnh tải lên."""
    runtime = request.app.state.runtime
    maximum = (
        runtime.config.visual_embedding.batch_size
        if source == "visual"
        else 64
    )
    identifiers, decoded = _decode_images(item_ids, images, maximum=maximum)
    started = perf_counter()
    try:
        vectors = np.asarray(runtime.embed_images(decoded, source=source))
        if vectors.ndim != 2 or vectors.shape[0] != len(identifiers):
            raise ValueError("image encoder returned the wrong result shape")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("image encoder returned non-finite vectors")
        status = _model_status(runtime, "dino" if source == "dino" else "visual_embedding")
    except Exception as error:
        raise _unavailable("Image embedding inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return EmbeddingResponse(
        model=status.checkpoint or source,
        revision=status.revision,
        dimension=int(vectors.shape[1]),
        normalized=True,
        item_ids=identifiers,
        embeddings=vectors.tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def ocr(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> OCRResponse:
    """Endpoint trích xuất văn bản (OCR) từ danh sách hình ảnh."""
    identifiers, decoded = _decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        values = runtime.ocr(decoded)
        if len(values) != len(identifiers):
            raise ValueError("OCR returned the wrong result count")
        status = _model_status(runtime, "ocr")
    except Exception as error:
        raise _unavailable("OCR inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return OCRResponse(
        model=status.checkpoint or "ocr",
        revision=status.revision,
        items=[
            OCRItem(
                item_id=item_id,
                text=value.text,
                raw_output=json_safe_ocr_raw(value.raw_output),
                regions=[
                    OCRRegionItem(
                        text=region.text,
                        confidence=region.confidence,
                        x_min=region.x_min,
                        y_min=region.y_min,
                        x_max=region.x_max,
                        y_max=region.y_max,
                    )
                    for region in value.regions
                ],
            )
            for item_id, value in zip(identifiers, values)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def shot_scores(
    request: Request,
    request_id: str = Form(min_length=1),
    tensor: UploadFile = File(),
) -> BoundaryScoreResponse:
    """Tính điểm Shot Boundary (cắt cảnh) từ tensor numpy."""
    return await _boundary_scores(request, request_id, tensor, source="shot")


async def event_scores(
    request: Request,
    request_id: str = Form(min_length=1),
    tensor: UploadFile = File(),
) -> BoundaryScoreResponse:
    """Tính điểm Event Boundary (chuyển hành động) từ tensor numpy."""
    return await _boundary_scores(request, request_id, tensor, source="event")


async def _boundary_scores(
    request: Request,
    request_id: str,
    tensor: UploadFile,
    *,
    source: str,
) -> BoundaryScoreResponse:
    """Xử lý chung để tính boundary scores từ dữ liệu dạng binary np.ndarray."""
    frames = _decode_tensor(tensor)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        values = np.asarray(runtime.boundary_scores(frames, source=source)).reshape(-1)
        if len(values) != len(frames) or not np.all(np.isfinite(values)):
            raise ValueError("boundary scorer returned invalid scores")
        status = _model_status(runtime, "transnet" if source == "shot" else "efficientgebd")
    except Exception as error:
        raise _unavailable("Boundary inference failed", error) from error
    return BoundaryScoreResponse(
        request_id=request_id,
        model=status.checkpoint or source,
        revision=status.revision,
        scores=values.astype(float).tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def transcribe(
    payload: AudioReferenceRequest, request: Request
) -> TranscriptInferenceResponse:
    """Thực thi mô hình ASR để trích xuất lời thoại từ tham chiếu Audio."""
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        segments = runtime.transcribe_reference(payload)
        status = _model_status(runtime, "asr")
    except Exception as error:
        raise _unavailable("ASR inference failed", error) from error
    return TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model=status.checkpoint or "asr",
        revision=status.revision,
        segments=segments,
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def diarize(
    payload: DiarizationRequest, request: Request
) -> TranscriptInferenceResponse:
    """Phân tách người nói (Diarization) từ Audio và Transcript có sẵn."""
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        segments = runtime.diarize_reference(payload)
        status = _model_status(runtime, "diarization")
    except Exception as error:
        raise _unavailable("Diarization inference failed", error) from error
    return TranscriptInferenceResponse(
        request_id=payload.request_id,
        video_id=payload.video_id,
        model=status.checkpoint or "diarization",
        revision=status.revision,
        segments=segments,
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def caption(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> CaptionResponse:
    identifiers, decoded = _decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        captions = runtime.caption(decoded)
        if len(captions) != len(identifiers):
            raise ValueError("captioner returned the wrong result count")
        if any(not value for value in captions):
            raise ValueError("captioner returned an empty caption")
    except Exception as error:
        raise _unavailable("Caption inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return CaptionResponse(
        model=runtime.config.caption_generation.model_checkpoint,
        revision=runtime.captioner.resolved_revision,
        items=[
            CaptionItem(item_id=item_id, caption=value)
            for item_id, value in zip(identifiers, captions)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )


async def rerank(
    request: Request,
    query: str = Form(min_length=1),
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> RerankResponse:
    identifiers, decoded = _decode_images(item_ids, images, maximum=100)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        scores = runtime.rerank(query, decoded)
        if len(scores) != len(identifiers):
            raise ValueError("reranker returned the wrong score count")
    except Exception as error:
        raise _unavailable("Reranking failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return RerankResponse(
        model=runtime.config.reranker.checkpoint,
        revision=getattr(runtime.reranker, "resolved_revision", None),
        items=[
            RerankItem(item_id=item_id, score=score)
            for item_id, score in zip(identifiers, scores)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )


def _decode_images(
    item_ids: str, uploads: list[UploadFile], *, maximum: int
) -> tuple[list[str], list[Image.Image]]:
    try:
        identifiers = json.loads(item_ids)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="item_ids must be JSON") from error
    if not isinstance(identifiers, list) or len(identifiers) != len(uploads):
        raise HTTPException(status_code=400, detail="item/image count mismatch")
    if not identifiers or len(identifiers) > maximum:
        raise HTTPException(
            status_code=400, detail=f"image batch must contain 1..{maximum}"
        )
    identifiers = [str(value).strip() for value in identifiers]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise HTTPException(status_code=400, detail="item_ids must be unique strings")
    try:
        payloads = [upload.file.read(5_000_001) for upload in uploads]
        if any(len(value) > 5_000_000 for value in payloads):
            raise ValueError("candidate image exceeds 5 MB")
        decoded = [Image.open(io.BytesIO(value)).convert("RGB") for value in payloads]
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid candidate image") from error
    return identifiers, decoded


def _decode_tensor(upload: UploadFile) -> np.ndarray:
    try:
        payload = upload.file.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("tensor exceeds 64 MiB")
        value = np.load(io.BytesIO(payload), allow_pickle=False)
        if value.ndim != 4 or value.shape[0] == 0 or value.shape[-1] != 3:
            raise ValueError("tensor must have shape [T,H,W,3]")
        if value.dtype not in {np.dtype("uint8"), np.dtype("float32")}:
            raise ValueError("tensor must use uint8 or float32")
        if value.dtype == np.float32 and not np.all(np.isfinite(value)):
            raise ValueError("tensor contains non-finite values")
        return np.ascontiguousarray(value)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid frame tensor") from error


def _model_status(runtime: object, name: str):
    status = runtime.readiness().models.get(name)
    if status is None or not status.loaded:
        raise RuntimeError(f"{name} model is not ready")
    return status


def _unavailable(prefix: str, error: Exception) -> HTTPException:
    detail = (str(error).strip() or type(error).__name__)[:160]
    return HTTPException(status_code=503, detail=f"{prefix}: {detail}")


# Lazy default app — only constructed when this attribute is accessed
# (e.g. `uvicorn thundercompute.server.api:app`), NOT at import time.
# Importing only `create_llm_app` from this module will NOT trigger LLMService
# construction, avoiding the BGE → sentence_transformers → torchcodec chain
# on environments where those libraries are unavailable.
def __getattr__(name: str):
    if name == "app":
        import sys as _sys
        _mod = _sys.modules[__name__]
        _app = create_llm_app()
        # Cache so subsequent accesses return the same object
        setattr(_mod, "app", _app)
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
