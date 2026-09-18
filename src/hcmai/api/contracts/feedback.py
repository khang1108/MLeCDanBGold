"""Contracts for KIS chat feedback, turn requests, and state responses."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hcmai.api.contracts.event_trail import EventTrailStateResponse
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.kis.models import EventId, KISIntent

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class RetrievalOverride(BaseModel):
    """Retrieval text overrides for dense and/or lexical retrieval for one event."""

    model_config = ConfigDict(extra="forbid")
    dense_text: str | None = None
    bm25_text: str | None = None


class FeedbackOpenRequest(BaseModel):
    """Payload to open a bounded feedback session from an existing search result."""

    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    original_query: NonBlank
    evidence_snapshot_id: NonBlank
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = Field(default=20, ge=1)
    initial_results: list[KISSearchResult] = Field(default_factory=list)


class FeedbackTurnRequest(BaseModel):
    """One feedback interaction turn initiated by user chat or shortcut button."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonBlank
    expected_feedback_revision: int = Field(ge=0)
    expected_kis_revision: int = Field(ge=0)
    expected_trail_revision: int | None = None
    message: NonBlank
    selected_result_id: str | None = None
    selected_event_id: str | None = None
    selected_frame_id: str | None = None


class FeedbackUndoRequest(BaseModel):
    """Request to revert the most recent committed feedback turn."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonBlank
    expected_feedback_revision: int = Field(ge=0)


class FeedbackStateResponse(BaseModel):
    """Complete state envelope returned by feedback open, turn, and undo operations."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: Literal["applied", "clarification", "exhausted", "proposal"]
    feedback_revision: int = Field(ge=1)
    intent: KISIntent
    retrieval_overrides: dict[str, RetrievalOverride] = Field(default_factory=dict)
    results: list[KISSearchResult] = Field(default_factory=list)
    evidence_snapshot_id: str
    trail: EventTrailStateResponse | None = None
    assistant_message: str | None = None
    query_proposal: dict[str, Any] | None = None
    changed_event_ids: list[str] = Field(default_factory=list)
    scope: str = "all_videos"
    can_undo: bool = False
    latency: dict[str, Any] | None = None


class FeedbackOpenResponse(BaseModel):
    """Response returned when opening a feedback session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    feedback_revision: int
    state: FeedbackStateResponse


__all__ = [
    "FeedbackOpenRequest",
    "FeedbackOpenResponse",
    "FeedbackStateResponse",
    "FeedbackTurnRequest",
    "FeedbackUndoRequest",
    "RetrievalOverride",
]
