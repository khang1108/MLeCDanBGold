"""Text and image embedding routes for hosted inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
import numpy as np

from hcmai.retrieval.embedding.inference_contracts import (
    EmbeddingResponse,
    TextEmbeddingResponse,
)
from llm.contracts import TextEmbeddingRequest
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(prefix="/v1/embeddings", tags=["embeddings"])


@router.post("/text", response_model=TextEmbeddingResponse)
async def embed_text(
    payload: TextEmbeddingRequest,
    request: Request,
) -> TextEmbeddingResponse:
    """Embed one bounded text batch in its requested encoder space."""

    started = perf_counter()
    runtime = runtime_from(request)
    config = (
        runtime.config.caption_embedding
        if payload.source == "text"
        else runtime.config.visual_embedding
    )
    if len(payload.texts) > config.batch_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"text batch must contain 1..{config.batch_size} items "
                f"for source {payload.source!r}"
            ),
        )
    try:
        vectors = runtime.embed_text(list(payload.texts), payload.source)
    except Exception as error:
        raise unavailable("Embedding inference failed", error) from error
    return TextEmbeddingResponse(
        model=config.model_name,
        revision=config.revision,
        dimension=int(vectors.shape[1]),
        normalized=True,
        embeddings=vectors.tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )


@router.post("/images", response_model=EmbeddingResponse)
async def embed_images(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> EmbeddingResponse:
    """Embed an aligned image batch with the configured visual encoder."""

    return await _embed_image_batch(request, item_ids, images, source="visual")


@router.post("/dino", response_model=EmbeddingResponse)
async def embed_dino(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> EmbeddingResponse:
    """Embed an aligned image batch with the configured DINO encoder."""

    return await _embed_image_batch(request, item_ids, images, source="dino")


async def _embed_image_batch(
    request: Request,
    item_ids: str,
    images: list[UploadFile],
    *,
    source: str,
) -> EmbeddingResponse:
    """Decode and embed one image batch without changing supplied item IDs."""

    runtime = runtime_from(request)
    maximum = runtime.config.visual_embedding.batch_size if source == "visual" else 64
    identifiers, decoded = decode_images(item_ids, images, maximum=maximum)
    started = perf_counter()
    try:
        vectors = np.asarray(runtime.embed_images(decoded, source=source))
        if vectors.ndim != 2 or vectors.shape[0] != len(identifiers):
            raise ValueError("image encoder returned the wrong result shape")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("image encoder returned non-finite vectors")
        model = "dino" if source == "dino" else "visual_embedding"
        model_status = loaded_model_status(runtime, model)
    except Exception as error:
        raise unavailable("Image embedding inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return EmbeddingResponse(
        model=model_status.checkpoint or source,
        revision=model_status.revision,
        dimension=int(vectors.shape[1]),
        normalized=True,
        item_ids=identifiers,
        embeddings=vectors.tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )
