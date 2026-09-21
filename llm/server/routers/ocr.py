"""GPU OCR route."""
from __future__ import annotations
from time import perf_counter
from fastapi import APIRouter, File, Form, Request, UploadFile
from llm.contracts import OCRItem, OCRRegionItem, OCRResponse
from offline.enrichment.ocr.models.entities import json_safe_ocr_raw
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(tags=["ocr"])

@router.post("/v1/enrichment/ocr", response_model=OCRResponse)
async def ocr(request: Request, item_ids: str = Form(), images: list[UploadFile] = File()) -> OCRResponse:
    identifiers, decoded = decode_images(item_ids, images, maximum=128)
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
                regions=[OCRRegionItem(text=r.text, confidence=r.confidence, x_min=r.x_min, y_min=r.y_min, x_max=r.x_max, y_max=r.y_max) for r in value.regions],
            )
            for item_id, value in zip(identifiers, values)
        ],
        latency_ms=(perf_counter() - started) * 1_000,
    )
