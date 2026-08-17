"""Private GPU inference API exposed only through Cloudflare Access."""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import numpy as np
from PIL import Image

from hcmai.common.schemas import (
    AudioReferenceRequest,
    BoundaryScoreResponse,
    CaptionItem,
    CaptionResponse,
    DiarizationRequest,
    EmbeddingResponse,
    InferenceReadiness,
    OCRItem,
    OCRResponse,
    RerankItem,
    RerankResponse,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
    TranscriptInferenceResponse,
    VQAInferenceEvidence,
    VQAInferenceResponse,
)
from hcmai.llm.pipeline import LLMService


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
        "/v1/vqa", vqa, methods=["POST"], response_model=VQAInferenceResponse
    )
    app.add_api_route(
        "/v1/vqa/multi",
        vqa_multi,
        methods=["POST"],
        response_model=VQAInferenceResponse,
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
    identifiers, decoded = _decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = request.app.state.runtime
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
            OCRItem(item_id=item_id, text=str(value))
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


async def vqa(
    request: Request,
    request_id: str = Form(min_length=1),
    frame_id: str = Form(min_length=1),
    video_id: str = Form(min_length=1),
    scene_context: str = Form(default="", max_length=1_000),
    question: str = Form(min_length=1, max_length=1_000),
    evidence: str = Form(default="{}"),
    image: UploadFile = File(),
) -> VQAInferenceResponse:
    try:
        context = VQAInferenceEvidence.model_validate_json(evidence)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid VQA evidence") from error
    _, decoded = _decode_images(json.dumps([frame_id]), [image], maximum=1)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        answer = runtime.answer_vqa(
            question, decoded[0], context, scene_context=scene_context
        )
    except Exception as error:
        raise _unavailable("VQA inference failed", error) from error
    finally:
        decoded[0].close()
    return VQAInferenceResponse(
        request_id=request_id,
        video_id=video_id,
        frame_ids=[frame_id],
        selected_frame_id=frame_id,
        question=question,
        answer=answer,
        answerable=True,
        grounded=True,
        confidence=0.5,
        model_name=runtime.config.vqa_model.checkpoint,
        latency_ms=max(0, int((perf_counter() - started) * 1_000)),
        evidence=context,
    )


async def vqa_multi(
    request: Request,
    request_id: str = Form(min_length=1),
    video_id: str = Form(min_length=1),
    frame_ids: str = Form(min_length=1),
    scene_context: str = Form(default="", max_length=1_000),
    question: str = Form(min_length=1, max_length=1_000),
    evidence: str = Form(default="{}"),
    images: list[UploadFile] = File(),
) -> VQAInferenceResponse:
    try:
        context = VQAInferenceEvidence.model_validate_json(evidence)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid VQA evidence") from error
    identifiers, decoded = _decode_images(frame_ids, images, maximum=32)
    started = perf_counter()
    runtime = request.app.state.runtime
    try:
        result = runtime.answer_vqa_multi(
            question,
            decoded,
            identifiers,
            context,
            scene_context=scene_context,
        )
    except Exception as error:
        raise _unavailable("Multi-frame VQA inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return VQAInferenceResponse(
        request_id=request_id,
        video_id=video_id,
        frame_ids=identifiers,
        selected_frame_id=result["selected_frame_id"],
        question=question,
        answer=result["answer"],
        answerable=result.get("answerable", True),
        grounded=True,
        confidence=result.get("confidence", 0.5),
        model_name=runtime.config.vqa_model.checkpoint,
        latency_ms=max(0, int((perf_counter() - started) * 1_000)),
        evidence=context,
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


app = create_llm_app()
