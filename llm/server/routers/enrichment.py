"""Caption and OCR enrichment routes for hosted inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, Form, Request, UploadFile

from offline.enrichment.inference_contracts import (
    CaptionItem,
    CaptionResponse,
    OCRItem,
    OCRRegionItem,
    OCRResponse,
)
from offline.enrichment.ocr.models.entities import json_safe_ocr_raw
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(tags=["enrichment"])


@router.post("/v1/captions", response_model=CaptionResponse)
async def caption(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> CaptionResponse:
    """Generate one non-empty caption for every supplied image ID."""

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
        items=[
            CaptionItem(item_id=item_id, caption=value)
            for item_id, value in zip(identifiers, captions)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )


@router.post("/v1/enrichment/ocr", response_model=OCRResponse)
async def ocr(
    request: Request,
    item_ids: str = Form(),
    images: list[UploadFile] = File(),
) -> OCRResponse:
    """Extract structured OCR evidence for every supplied image ID."""

    identifiers, decoded = decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        values = runtime.ocr(decoded)
        if len(values) != len(identifiers):
            raise ValueError("OCR returned the wrong result count")
        model_status = loaded_model_status(runtime, "ocr")
    except Exception as error:
        raise unavailable("OCR inference failed", error) from error
    finally:
        for image in decoded:
            image.close()
    return OCRResponse(
        model=model_status.checkpoint or "ocr",
        revision=model_status.revision,
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
