"""HTTP-only contracts for local temporal exploration branches.

These models serialize immutable exploration snapshots. They do not own
temporal scoring, alignment, canonical identities, or branch state.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from hcmai.temporal.planner import normalize_event_texts

_NonBlankString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
_Milliseconds = Annotated[int, Field(strict=True, ge=0)]
_Interval = tuple[_Milliseconds, _Milliseconds]


class ExplorationOpenRequest(BaseModel):
    """Capture one immutable retrieval snapshot for a selected video."""

    model_config = ConfigDict(extra="forbid")

    query: _NonBlankString
    events: list[_NonBlankString] = Field(min_length=1)
    retrieval_events: list[_NonBlankString] = Field(min_length=1)
    caption_events: list[_NonBlankString] | None
    use_dense: bool
    use_bm25: bool
    video_id: _NonBlankString
    window: _Interval

    @field_validator("events", "retrieval_events", "caption_events")
    @classmethod
    def normalize_events(
        cls,
        events: list[str] | None,
    ) -> list[str] | None:
        """Normalize whitespace before binding event positions to the core."""

        return list(normalize_event_texts(events)) if events is not None else None

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ExplorationOpenRequest":
        """Keep event rows aligned and require an active retrieval source."""

        if not self.use_dense and not self.use_bm25:
            raise ValueError("at least one of use_dense or use_bm25 must be true")
        if len(self.retrieval_events) != len(self.events):
            raise ValueError("retrieval_events must match the event count")
        if (
            self.caption_events is not None
            and len(self.caption_events) != len(self.events)
        ):
            raise ValueError("caption_events must match the event count")
        if self.window[0] > self.window[1]:
            raise ValueError("window start_ms must not exceed end_ms")
        return self


class ExplorationActionRequest(BaseModel):
    """Apply one revision-guarded action to a local exploration branch."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: Annotated[int, Field(strict=True, ge=1)]
    event_version: UUID
    scoring_revision: UUID
    action: Literal["confirm", "reject", "window", "undo"]
    event_index: Annotated[int, Field(strict=True, ge=0)] | None = None
    interval: _Interval | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ExplorationActionRequest":
        """Reject malformed action shapes before the immutable core is called."""

        if self.interval is not None and self.interval[0] > self.interval[1]:
            raise ValueError("interval start_ms must not exceed end_ms")
        if self.action in {"confirm", "reject"}:
            if self.event_index is None:
                raise ValueError(f"{self.action} action requires event_index")
            if self.interval is None:
                raise ValueError(f"{self.action} action requires interval")
        elif self.action == "window":
            if self.event_index is not None:
                raise ValueError("window action does not accept event_index")
            if self.interval is None:
                raise ValueError("window action requires interval")
        elif self.event_index is not None or self.interval is not None:
            raise ValueError("undo action does not accept event_index or interval")
        return self


class ExplorationConditionsResponse(BaseModel):
    """Serialize per-event temporal constraints without changing their meaning."""

    model_config = ConfigDict(extra="forbid")

    window: _Interval
    confirmed: list[_Interval | None]
    rejected: list[list[_Interval]]


class ExplorationPathResponse(BaseModel):
    """Serialize one canonical path produced by the shared temporal decoder."""

    model_config = ConfigDict(extra="forbid")

    video_id: str
    score: float
    frame_ids: list[str]
    frame_idxs: list[int]
    timestamps_ms: list[int]


class ExplorationViewResponse(BaseModel):
    """Expose one immutable branch view and its canonical aligned paths."""

    model_config = ConfigDict(extra="forbid")

    revision: int
    event_version: str
    video_id: str
    conditions: ExplorationConditionsResponse
    status: Literal[
        "ok",
        "contradictory_conditions",
        "no_indexed_frames",
        "no_valid_path",
    ]
    paths: list[ExplorationPathResponse]
    changed_event_indices: list[int]
    comparison_available: bool
    can_undo: bool


class ExplorationEnvelope(BaseModel):
    """Return the stable handle, scoring generation, and branch snapshot."""

    model_config = ConfigDict(extra="forbid")

    handle: str
    scoring_revision: str
    view: ExplorationViewResponse
