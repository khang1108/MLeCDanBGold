"""Structured query translation and candidate-generation routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from llm.contracts import (
    QueryCandidatesRequest,
    QueryCandidatesResponse,
    QueryEventsRequest,
    QueryTranslationResponse,
)
from llm.server.dependencies import runtime_from

router = APIRouter(prefix="/query-preparation", tags=["query-preparation"])


@router.post("/translate", response_model=QueryTranslationResponse)
async def translate_query_events(
    payload: QueryEventsRequest,
    request: Request,
) -> QueryTranslationResponse:
    """Translate ordered events while preserving their count and order."""

    try:
        events = runtime_from(request).translate_query_events(list(payload.events))
        response = QueryTranslationResponse(events=events)
        if len(response.events) != len(payload.events):
            raise ValueError("translation changed event count")
        return response
    except HTTPException:
        raise
    except Exception as error:
        detail = (str(error) or type(error).__name__)[:160]
        raise HTTPException(
            status_code=502,
            detail=f"Query translation failed: {detail}",
        ) from error


@router.post("/candidates", response_model=QueryCandidatesResponse)
async def generate_query_candidates(
    payload: QueryCandidatesRequest,
    request: Request,
) -> QueryCandidatesResponse:
    """Generate exactly five candidate bundles aligned to request events."""

    try:
        value = runtime_from(request).generate_query_candidates(
            list(payload.events),
            payload.candidate_count,
        )
        response = QueryCandidatesResponse.model_validate(value)
        expected = len(payload.events)
        if len(response.literal_en) != expected or any(
            len(candidate) != expected for candidate in response.candidates
        ):
            raise ValueError("candidate generation changed event count")
        return response
    except HTTPException:
        raise
    except Exception as error:
        detail = (str(error) or type(error).__name__)[:160]
        raise HTTPException(
            status_code=502,
            detail=f"Query candidate generation failed: {detail}",
        ) from error
