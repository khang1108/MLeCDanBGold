"""GPU caption route."""
from __future__ import annotations
from time import perf_counter
from fastapi import APIRouter, File, Form, Request, UploadFile
from llm.contracts import CaptionItem, CaptionResponse
from llm.server.dependencies import runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(tags=["captions"])

@router.post("/v1/captions", response_model=CaptionResponse)
async def caption(request: Request, item_ids: str = Form(), images: list[UploadFile] = File()) -> CaptionResponse:
    identifiers, decoded = decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        captions = runtime.caption(decoded)
        if len(captions) != len(identifiers):
            raise ValueError("captioner returned the wrong result count")
        if any(not value for value in captions):
            raise ValueError("captioner returned an empty caption")
    except Exception as error:
        raise unavailable("Caption inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return CaptionResponse(
        model=runtime.config.caption_generation.model_checkpoint,
        revision=runtime.captioner.resolved_revision,
        items=[CaptionItem(item_id=item_id, caption=value) for item_id, value in zip(identifiers, captions)],
        latency_ms=(perf_counter() - started) * 1_000,
    )
