"""Bounded visual reranking route for hosted inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, Form, Request, UploadFile

from llm.contracts import RerankItem, RerankResponse
from llm.server.dependencies import runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(tags=["reranking"])


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    request: Request,
    query: str = Form(min_length=1),
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> RerankResponse:
    """Score a bounded image candidate set without changing item identity."""

    identifiers, decoded = decode_images(item_ids, images, maximum=100)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        scores = runtime.rerank(query, decoded)
        if len(scores) != len(identifiers):
            raise ValueError("reranker returned the wrong score count")
    except Exception as error:
        raise unavailable("Reranking failed", error) from error
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
