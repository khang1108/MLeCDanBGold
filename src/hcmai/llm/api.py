"""Private GPU inference API exposed only through Cloudflare Access."""

from __future__ import annotations

import io
import json
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from hcmai.common.schemas import (
    CaptionItem,
    CaptionResponse,
    ConversationInferenceRequest,
    ConversationState,
    InferenceReadiness,
    RerankItem,
    RerankResponse,
    TextEmbeddingRequest,
    TextEmbeddingResponse,
)
from hcmai.llm.runtime import LLMRuntime


def create_llm_app(runtime: LLMRuntime | None = None) -> FastAPI:
    """Create an injectable inference API; production models load in lifespan."""
    owned = runtime or LLMRuntime.from_environment()
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
        "/ready", ready, methods=["GET"], response_model=InferenceReadiness
    )
    app.add_api_route(
        "/v1/embeddings/text",
        embed,
        methods=["POST"],
        response_model=TextEmbeddingResponse,
    )
    app.add_api_route(
        "/v1/rerank", rerank, methods=["POST"], response_model=RerankResponse
    )
    app.add_api_route(
        "/v1/conversation/resolve",
        resolve,
        methods=["POST"],
        response_model=ConversationState,
    )
    return app


def _lifespan(runtime: LLMRuntime):
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
    try:
        vectors = runtime.embed_text(list(payload.texts), payload.source)
    except Exception as error:
        raise _unavailable("Embedding inference failed", error) from error
    return TextEmbeddingResponse(
        model=getattr(runtime.config, f"{payload.source}_embedding").model_name,
        dimension=int(vectors.shape[1]),
        normalized=True,
        embeddings=vectors.tolist(),
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


async def resolve(
    payload: ConversationInferenceRequest, request: Request
) -> ConversationState:
    try:
        output = request.app.state.runtime.resolve(payload.model_dump(mode="json"))
        return ConversationState.model_validate(output)
    except Exception as error:
        raise _unavailable("Conversation inference failed", error) from error


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


def _unavailable(prefix: str, error: Exception) -> HTTPException:
    detail = (str(error).strip() or type(error).__name__)[:160]
    return HTTPException(status_code=503, detail=f"{prefix}: {detail}")


app = create_llm_app()
