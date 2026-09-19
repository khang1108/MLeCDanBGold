"""GPU object-detection route."""
from __future__ import annotations
from time import perf_counter
from fastapi import APIRouter, File, Form, Request, UploadFile
from llm.contracts import ObjectItem, ObjectResponse
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable
from llm.server.parsing import decode_images

router = APIRouter(tags=["objects"])

@router.post("/v1/enrichment/objects", response_model=ObjectResponse)
async def objects(
    request: Request,
    item_ids: str = Form(),
    min_confidence: float = Form(default=0.20),
    top_k: int = Form(default=30),
    images: list[UploadFile] = File(),
) -> ObjectResponse:
    identifiers, decoded = decode_images(item_ids, images, maximum=128)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        values = runtime.objects(decoded, min_confidence=min_confidence, top_k=top_k)
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
        items=[ObjectItem(item_id=item_id, labels=list(value["detection_class_entities"]), scores=list(value["detection_scores"]), boxes=list(value["detection_boxes"])) for item_id, value in zip(identifiers, values, strict=True)],
        latency_ms=(perf_counter() - started) * 1_000,
    )
