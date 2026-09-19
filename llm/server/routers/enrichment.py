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
    ObjectItem,
    ObjectResponse,
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

@router.post("/v1/enrichment/objects", response_model=ObjectResponse)
async def objects(
    request: Request,
    item_ids: str = Form(),
    min_confidence: float = Form(default=0.20),
    top_k: int = Form(default=30),
    images: list[UploadFile] = File(),
) -> ObjectResponse:
    """Detect objects in an ordered image batch using the hosted YOLOE model."""

    identifiers, decoded = decode_images(item_ids, images, maximum=64)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        values = runtime.objects(
            decoded, min_confidence=min_confidence, top_k=top_k
        )
        if len(values) != len(identifiers):
            raise ValueError("object detector returned the wrong result count")
        model_status = loaded_model_status(runtime, "objects")
    except Exception as error:
        raise unavailable("Object detection inference failed", error) from error
    finally:
        for image in decoded:
            image.close()

    return ObjectResponse(
        model=model_status.checkpoint or "objects",
        revision=model_status.revision,
        items=[
            ObjectItem(
                item_id=item_id,
                labels=list(value["detection_class_entities"]),
                scores=list(value["detection_scores"]),
                boxes=list(value["detection_boxes"]),
            )
            for item_id, value in zip(identifiers, values, strict=True)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )
