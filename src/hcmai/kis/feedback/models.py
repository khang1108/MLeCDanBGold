"""Domain models for KIS chat feedback actions, contexts, sessions, and checkpoints."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hcmai.api.contracts.feedback import (
    FeedbackStateResponse,
    RetrievalOverride,
)
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.kis.models import EventId, KISIntent


class RefineRetrievalAction(BaseModel):
    """Update retrieval query text without modifying canonical semantic intent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["refine_retrieval"] = "refine_retrieval"
    event_ids: list[str] = Field(min_length=1)
    refinements: dict[str, str] = Field(
        description="Mapping from event_id to retrieval search description"
    )


class RepairEventAction(BaseModel):
    """Trigger search repair for an event within current scope."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["repair_event"] = "repair_event"
    event_id: str


class ClarifyAction(BaseModel):
    """Ask a short clarifying question when request is ambiguous or underspecified."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["clarify"] = "clarify"
    question: str


class QueryEditProposalAction(BaseModel):
    """Propose an action on the query hypothesis without committing directly."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["query_edit_proposal"] = "query_edit_proposal"
    action: dict[str, Any]
    explanation: str


FeedbackAction = Annotated[
    QueryEditProposalAction
    | RefineRetrievalAction
    | RepairEventAction
    | ClarifyAction,
    Field(discriminator="type"),
]


class FeedbackResolution(BaseModel):
    """Structured resolution envelope produced by the feedback resolver."""

    model_config = ConfigDict(extra="forbid")

    action: FeedbackAction

    @model_validator(mode="before")
    @classmethod
    def _ensure_action_type(cls, data: Any) -> Any:
        if isinstance(data, dict):
            act = data.get("action")
            if isinstance(act, dict) and "type" not in act:
                if "explanation" in act and "action" in act:
                    act["type"] = "query_edit_proposal"
                elif "refinements" in act:
                    act["type"] = "refine_retrieval"
                elif "question" in act:
                    act["type"] = "clarify"
                elif "event_id" in act:
                    act["type"] = "repair_event"
        return data


class FeedbackChatTurn(BaseModel):
    """One chat transcript record in the feedback session history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    message: str
    action_type: str | None = None
    timestamp_ms: float = 0.0


class FeedbackResolveContext(BaseModel):
    """Context passed to the bounded LLM feedback resolver."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    intent: KISIntent
    retrieval_overrides: dict[str, RetrievalOverride] = Field(default_factory=dict)
    selected_result_id: str | None = None
    selected_event_id: str | None = None
    selected_frame_id: str | None = None
    anchors: dict[str, str] = Field(
        default_factory=dict,
        description="Event ID to confirmed anchored frame ID",
    )
    recent_turns: list[FeedbackChatTurn] = Field(
        default_factory=list,
        max_length=4,
        description="Up to 4 recent chat transcript turns",
    )
    scope: str = "all_videos"


class FeedbackCheckpoint(BaseModel):
    """Immutable state snapshot for unified feedback Undo."""

    model_config = ConfigDict(extra="forbid")

    intent: KISIntent
    retrieval_overrides: dict[str, RetrievalOverride]
    results: list[KISSearchResult]
    evidence_snapshot_id: str
    scope: str
    exclusions: tuple[tuple[str, str, str], ...] = ()
    trail_session_id: str | None = None
    assistant_message: str | None = None
    changed_event_ids: tuple[str, ...] = ()


class FeedbackSession(BaseModel):
    """Active server session owning feedback revision, chat turns, and undo checkpoints."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    original_query: str
    use_dense: bool = True
    use_bm25: bool = True
    top_k: int = 20
    feedback_revision: int = 1
    state: FeedbackStateResponse
    history: tuple[FeedbackCheckpoint, ...] = ()
    chat_turns: tuple[FeedbackChatTurn, ...] = ()
    exclusions: tuple[tuple[str, str, str], ...] = ()
    committed_requests: dict[str, tuple[dict[str, Any], FeedbackStateResponse]] = Field(
        default_factory=dict
    )
    created_at: float = 0.0
    updated_at: float = 0.0


__all__ = [
    "ClarifyAction",
    "FeedbackAction",
    "FeedbackChatTurn",
    "FeedbackCheckpoint",
    "FeedbackResolution",
    "FeedbackResolveContext",
    "FeedbackSession",
    "QueryEditProposalAction",
    "RefineRetrievalAction",
    "RepairEventAction",
]
