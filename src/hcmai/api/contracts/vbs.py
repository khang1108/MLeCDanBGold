"""Browser-safe session and DRES submission contracts.

Only VBS participant IDs and reviewed candidate revisions cross this API.
DRES credentials, sessions, and transport error bodies remain backend-only.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from hcmai.api.contracts.workspace import AnswerWorkspaceSnapshot


NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class VbsSessionConnectRequest(BaseModel):
    """Connect the configured backend credential mapped to this user ID."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank


class VbsSessionStatus(BaseModel):
    """Return session connectivity without a DRES token or username."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    connected: bool


class VbsCandidateRevision(BaseModel):
    """One frozen workspace candidate ID and optimistic revision."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: NonBlank
    expected_revision: int = Field(ge=1, strict=True)


class VbsSingleSubmissionRequest(BaseModel):
    """One reviewed KIS FRAME or VQA TEXT submission request."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    task_scope_key: NonBlank
    expected_workspace_revision: int = Field(ge=0, strict=True)
    candidate_id: NonBlank
    expected_revision: int = Field(ge=1, strict=True)


class VbsAvsSubmissionRequest(BaseModel):
    """One deterministic, complete AVS FRAME collection."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    task_scope_key: NonBlank
    expected_workspace_revision: int = Field(ge=0, strict=True)
    candidates: list[VbsCandidateRevision] = Field(min_length=1)


class VbsSubmissionResolutionRequest(BaseModel):
    """Operator-reviewed outcome for the currently reserved UNKNOWN attempt."""

    model_config = ConfigDict(extra="forbid")

    user_id: NonBlank
    outcome: Literal["accepted", "not_accepted"]


class VbsSubmissionResponse(BaseModel):
    """Safe status returned after an accepted, rejected, or ambiguous send."""

    model_config = ConfigDict(extra="forbid")

    attempt_id: NonBlank
    state: Literal["ACCEPTED", "NOT_ACCEPTED", "UNKNOWN"]
    accepted: bool | None
    verdict: Literal["CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"] | None = None
    message: str
    workspace: AnswerWorkspaceSnapshot


__all__ = [
    "VbsAvsSubmissionRequest",
    "VbsCandidateRevision",
    "VbsSessionConnectRequest",
    "VbsSessionStatus",
    "VbsSingleSubmissionRequest",
    "VbsSubmissionResolutionRequest",
    "VbsSubmissionResponse",
]
