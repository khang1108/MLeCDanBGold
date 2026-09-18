"""HTTP contracts for lossless query replay history and viewed-frame activity."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HistorySnapshot = dict[str, JsonValue]


class QueryOperationMetadata(BaseModel):
    """Metadata describing the semantic operation and revision that produced a history record."""

    model_config = ConfigDict(extra="forbid")

    semantic_revision: int = Field(ge=0)
    operation_kind: str
    affected_event_ids: list[str] = Field(default_factory=list)
    image_added: list[str] = Field(default_factory=list)
    image_removed: list[str] = Field(default_factory=list)
    search_only: bool = False
    action: str | None = None
    scope: str | None = None
    feedback_revision: int | None = Field(default=None, ge=0)


class QueryInteractionEventCreate(BaseModel):
    """Append-only interaction event payload emitted during user exploration."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["result_open", "submission"]
    semantic_revision: int = Field(ge=0)
    event_id: str | None = None
    frame_id: str | None = None
    video_id: str | None = None
    timestamp_ms: int | None = Field(default=None, ge=0)


class QueryInteractionEventRecord(BaseModel):
    """Persisted interaction event returned to the caller."""

    query_id: str
    sequence_id: int
    event_type: Literal["result_open", "submission"]
    semantic_revision: int
    event_id: str | None = None
    frame_id: str | None = None
    video_id: str | None = None
    timestamp_ms: int | None = None
    created_at: str


class QueryHistoryCreate(BaseModel):
    """Frontend-owned query data to persist after a successful search."""

    model_config = ConfigDict(extra="forbid")

    query_id: NonBlank
    user_id: NonBlank
    query_text: NonBlank
    result_snapshot: HistorySnapshot
    operation_metadata: QueryOperationMetadata | None = None


class QueryHistoryViewedFrameUpdate(BaseModel):
    """Request to record one canonical frame opened by the user."""

    model_config = ConfigDict(extra="forbid")

    frame_id: NonBlank


class FrameActivity(BaseModel):
    """Canonical frames viewed from one query's replay results."""

    viewed_frame_ids: list[str]


class QueryHistoryRecord(BaseModel):
    """Replayable query history returned to the frontend."""

    query_id: str
    query_text: str
    result_snapshot: HistorySnapshot
    frame_activity: FrameActivity
    operation_metadata: QueryOperationMetadata | None = None


class QueryHistoryList(BaseModel):
    """Newest query history items for one user."""

    items: list[QueryHistoryRecord]


__all__ = [
    "FrameActivity",
    "HistorySnapshot",
    "QueryHistoryCreate",
    "QueryHistoryList",
    "QueryHistoryRecord",
    "QueryHistoryViewedFrameUpdate",
    "QueryInteractionEventCreate",
    "QueryInteractionEventRecord",
    "QueryOperationMetadata",
]
