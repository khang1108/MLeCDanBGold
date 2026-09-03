"""Liveness and model-readiness routes for hosted inference."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from offline.enrichment.inference_contracts import InferenceReadiness
from llm.server.dependencies import runtime_from

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Report process liveness without forcing model readiness."""

    return {"status": "ok"}


@router.get("/ready", response_model=InferenceReadiness)
async def ready(request: Request) -> InferenceReadiness:
    """Report enabled-model readiness and checkpoint provenance."""

    value = runtime_from(request).readiness()
    if not value.ready:
        raise HTTPException(status_code=503, detail="Models are not ready")
    return value
