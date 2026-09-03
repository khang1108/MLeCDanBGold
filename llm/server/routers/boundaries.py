"""Shot and event boundary-scoring routes for hosted inference."""

from __future__ import annotations

from time import perf_counter

from fastapi import APIRouter, File, Form, Request, UploadFile
import numpy as np

from llm.contracts import BoundaryScoreResponse
from llm.server.dependencies import loaded_model_status, runtime_from, unavailable
from llm.server.parsing import decode_tensor

router = APIRouter(prefix="/v1/preprocessing", tags=["preprocessing"])


@router.post("/shot-scores", response_model=BoundaryScoreResponse)
async def shot_scores(
    request: Request,
    request_id: str = Form(min_length=1),
    tensor: UploadFile = File(),
) -> BoundaryScoreResponse:
    """Score shot boundaries for one ordered frame tensor."""

    return await _boundary_scores(request, request_id, tensor, source="shot")


@router.post(
    "/event-scores",
    response_model=BoundaryScoreResponse,
    include_in_schema=False,
)
@router.post("/event-window-scores", response_model=BoundaryScoreResponse)
async def event_scores(
    request: Request,
    request_id: str = Form(min_length=1),
    tensor: UploadFile = File(),
) -> BoundaryScoreResponse:
    """Score event boundaries for one ordered frame tensor."""

    return await _boundary_scores(request, request_id, tensor, source="event")


async def _boundary_scores(
    request: Request,
    request_id: str,
    tensor: UploadFile,
    *,
    source: str,
) -> BoundaryScoreResponse:
    """Validate a tensor and translate one boundary-model result."""

    frames = decode_tensor(tensor)
    started = perf_counter()
    runtime = runtime_from(request)
    try:
        values = np.asarray(runtime.boundary_scores(frames, source=source)).reshape(-1)
        if len(values) != len(frames) or not np.all(np.isfinite(values)):
            raise ValueError("boundary scorer returned invalid scores")
        model_name = "transnet" if source == "shot" else "efficientgebd"
        model_status = loaded_model_status(runtime, model_name)
    except Exception as error:
        raise unavailable("Boundary inference failed", error) from error
    return BoundaryScoreResponse(
        request_id=request_id,
        model=model_status.checkpoint or source,
        revision=model_status.revision,
        scores=values.astype(float).tolist(),
        latency_ms=(perf_counter() - started) * 1_000,
    )
