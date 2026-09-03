"""HTTP contracts for lossless replay history and shared submission files."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints


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


class QueryHistorySubmissionUpdate(BaseModel):
    """Request to associate committed submission data with one query."""

    model_config = ConfigDict(extra="forbid")

    submission_file_name: NonBlank
    submission_line: NonBlank
    frame_ids: list[NonBlank] = Field(min_length=1)


class FrameActivity(BaseModel):
    """Canonical frames viewed or submitted from one query."""

    viewed_frame_ids: list[str]
    submitted_frame_ids: list[str]


class QueryHistoryRecord(BaseModel):
    """Replayable query history returned to the frontend."""

    query_id: str
    query_text: str
    submission_files: list[str]
    result_snapshot: HistorySnapshot
    frame_activity: FrameActivity


class QueryHistoryList(BaseModel):
    """Newest query history items for one user."""

    items: list[QueryHistoryRecord]


class SubmissionFile(BaseModel):
    """One shared submission file and its optimistic-lock revision."""

    name: str
    content: str
    is_validated: bool
    revision: int


class SubmissionFileList(BaseModel):
    """Current shared submission files used to hydrate the workspace."""

    files: list[SubmissionFile]


class SubmissionFileCreate(BaseModel):
    """WebSocket command that creates one shared file."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["submission_file.create"]
    name: NonBlank
    content: str = ""


class SubmissionFileUpdate(BaseModel):
    """WebSocket command that replaces file content."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["submission_file.update"]
    name: NonBlank
    content: str
    expected_revision: int = Field(ge=1)


class SubmissionFileValidate(BaseModel):
    """WebSocket command that changes file validation state."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["submission_file.validate"]
    name: NonBlank
    is_validated: bool
    expected_revision: int = Field(ge=1)


class SubmissionFileDelete(BaseModel):
    """WebSocket command that deletes one shared file."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["submission_file.delete"]
    name: NonBlank
    expected_revision: int = Field(ge=1)


class SubmissionFileClear(BaseModel):
    """WebSocket command that clears all shared submission files."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["submission_file.clear"] = "submission_file.clear"


SubmissionFileCommand = (
    SubmissionFileCreate
    | SubmissionFileUpdate
    | SubmissionFileValidate
    | SubmissionFileDelete
    | SubmissionFileClear
)


__all__ = [
    "FrameActivity",
    "HistorySnapshot",
    "QueryHistoryCreate",
    "QueryHistoryList",
    "QueryHistoryRecord",
    "QueryHistorySubmissionUpdate",
    "QueryHistoryViewedFrameUpdate",
    "SubmissionFile",
    "SubmissionFileClear",
    "SubmissionFileCommand",
    "SubmissionFileCreate",
    "SubmissionFileDelete",
    "SubmissionFileList",
    "SubmissionFileUpdate",
    "SubmissionFileValidate",
]
