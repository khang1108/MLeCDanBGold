"""HTTP contracts for lossless query replay history and viewed-frame activity."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
HistorySnapshot = dict[str, JsonValue]


class QueryHistoryCreate(BaseModel):
    """Frontend-owned query data to persist after a successful search."""

    model_config = ConfigDict(extra="forbid")

    query_id: NonBlank
    user_id: NonBlank
    query_text: NonBlank
    result_snapshot: HistorySnapshot


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
]
