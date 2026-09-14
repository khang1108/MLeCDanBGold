"""Typed HTTP and WebSocket contracts for structured answer collaboration.

Workspace candidates preserve exact submitted identity and timestamps. DRES
credentials, session IDs, and mutable client-side frame-index coordinates are
not part of these contracts.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AnswerCandidate(BaseModel):
    """One typed answer row with canonical IDs and audit metadata."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: NonBlank
    kind: Literal["FRAME", "TEXT"]
    source_frame_id: str | None = None
    video_id: str | None = None
    timestamp_ms: int | None = Field(default=None, ge=0, strict=True)
    text: str | None = None
    contributed_by_user_id: NonBlank
    created_at_ms: int = Field(ge=0)
    revision: int = Field(ge=1)
    submitted_at_ms: int | None = Field(default=None, ge=0)
    submitted_by_user_id: str | None = None
    dres_status: str | None = None
    evaluation_id: NonBlank
    task_scope_key: NonBlank

    @model_validator(mode="after")
    def validate_candidate_shape(self) -> "AnswerCandidate":
        """Keep FRAME coordinates and TEXT answers mutually exclusive."""

        if self.kind == "FRAME":
            if not self.video_id or not self.video_id.strip() or self.timestamp_ms is None or self.text is not None:
                raise ValueError("FRAME candidates require video_id and timestamp_ms only")
        elif self.text is None or self.video_id is not None or self.timestamp_ms is not None or self.source_frame_id is not None:
            raise ValueError("TEXT candidates require text and cannot contain frame fields")
        return self


class SubmissionAttemptSummary(BaseModel):
    """Redacted lock status for one durable submission attempt."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: NonBlank
    kind: Literal["KIS", "VQA", "AVS"]
    state: Literal["FORWARDING", "UNKNOWN"]
    submitted_by_user_id: NonBlank
    created_at_ms: int = Field(ge=0)


class AnswerWorkspaceSnapshot(BaseModel):
    """Current shared candidates plus stored and live task scope."""

    model_config = ConfigDict(extra="forbid")

    avs_enabled: bool
    evaluation_id: str | None = None
    task_scope_key: str | None = None
    task_name: str | None = None
    revision: int = Field(ge=0)
    updated_by_user_id: str
    updated_at_ms: int = Field(ge=0)
    candidates: list[AnswerCandidate] = Field(default_factory=list)
    pending_submission: SubmissionAttemptSummary | None = None
    active_evaluation_id: NonBlank
    active_task_scope_key: NonBlank
    active_task_name: NonBlank
    task_scope_mismatch: bool = False


class AnswerCandidateMutation(BaseModel):
    """Result of one mutation with its resulting workspace revision."""

    model_config = ConfigDict(extra="forbid")

    candidate: AnswerCandidate
    workspace_revision: int = Field(ge=0)


class AnswerModeMutation(BaseModel):
    """Result of a shared AVS-mode update without a fake candidate row."""

    model_config = ConfigDict(extra="forbid")

    avs_enabled: bool
    workspace_revision: int = Field(ge=0)


class AnswerWorkspaceCommandBase(BaseModel):
    """Common optimistic workspace revision sent over WebSocket."""

    model_config = ConfigDict(extra="forbid")

    expected_workspace_revision: int = Field(ge=0)


class AnswerAddFrame(AnswerWorkspaceCommandBase):
    """Add one exact FRAME moment, optionally retaining source provenance."""

    type: Literal["answer.add_frame"]
    video_id: NonBlank
    timestamp_ms: int = Field(ge=0, strict=True)
    source_frame_id: NonBlank | None = None


class AnswerAddText(AnswerWorkspaceCommandBase):
    """Add one plaintext VQA candidate."""

    type: Literal["answer.add_text"]
    text: NonBlank


class AnswerUpdateFrame(AnswerWorkspaceCommandBase):
    """Edit an existing temporal candidate with an exact new moment."""

    type: Literal["answer.update_frame"]
    candidate_id: NonBlank
    expected_candidate_revision: int = Field(ge=1)
    video_id: NonBlank
    timestamp_ms: int = Field(ge=0, strict=True)


class AnswerUpdateText(AnswerWorkspaceCommandBase):
    """Edit an existing VQA text candidate."""

    type: Literal["answer.update_text"]
    candidate_id: NonBlank
    expected_candidate_revision: int = Field(ge=1)
    text: NonBlank


class AnswerDelete(AnswerWorkspaceCommandBase):
    """Delete one candidate at a known candidate and workspace revision."""

    type: Literal["answer.delete"]
    candidate_id: NonBlank
    expected_candidate_revision: int = Field(ge=1)


class AnswerClear(AnswerWorkspaceCommandBase):
    """Clear candidates after the user confirms the destructive action."""

    type: Literal["answer.clear"]


class AnswerModeSet(AnswerWorkspaceCommandBase):
    """Set the shared AVS collection mode."""

    type: Literal["answer.mode.set"]
    avs_enabled: bool


class AnswerTaskClearAndSwitch(AnswerWorkspaceCommandBase):
    """Clear old-task candidates and bind to a server-verified new task."""

    type: Literal["answer.task.clear_and_switch"]
    expected_old_evaluation_id: NonBlank
    expected_old_task_scope_key: NonBlank
    target_evaluation_id: NonBlank
    target_task_scope_key: NonBlank
    target_task_name: NonBlank


AnswerWorkspaceCommand = Annotated[
    AnswerAddFrame
    | AnswerAddText
    | AnswerUpdateFrame
    | AnswerUpdateText
    | AnswerDelete
    | AnswerClear
    | AnswerModeSet
    | AnswerTaskClearAndSwitch,
    Field(discriminator="type"),
]


class AnswerWorkspaceEvent(BaseModel):
    """One committed workspace event broadcast with a full durable snapshot."""

    model_config = ConfigDict(extra="forbid")

    type: str
    workspace: AnswerWorkspaceSnapshot


__all__ = [
    "AnswerAddFrame",
    "AnswerAddText",
    "AnswerCandidate",
    "AnswerCandidateMutation",
    "AnswerClear",
    "AnswerDelete",
    "AnswerModeSet",
    "AnswerModeMutation",
    "AnswerTaskClearAndSwitch",
    "AnswerUpdateFrame",
    "AnswerUpdateText",
    "AnswerWorkspaceCommand",
    "AnswerWorkspaceEvent",
    "AnswerWorkspaceSnapshot",
    "SubmissionAttemptSummary",
]
