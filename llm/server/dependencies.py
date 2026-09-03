"""Shared runtime access and error translation for inference API routers.

This module owns transport-level dependency lookup only. Model lifecycle and
capability implementations remain behind :class:`llm.pipeline.LLMService`.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request


def runtime_from(request: Request) -> Any:
    """Return the process-owned inference runtime attached during app creation."""

    return request.app.state.runtime


def loaded_model_status(runtime: Any, name: str) -> Any:
    """Return one loaded readiness row or fail the capability request."""

    model_status = runtime.readiness().models.get(name)
    if model_status is None or not model_status.loaded:
        raise RuntimeError(f"{name} model is not ready")
    return model_status


def unavailable(prefix: str, error: Exception) -> HTTPException:
    """Translate an internal capability failure into a bounded HTTP 503 detail."""

    detail = (str(error).strip() or type(error).__name__)[:160]
    return HTTPException(status_code=503, detail=f"{prefix}: {detail}")
