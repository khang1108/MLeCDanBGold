"""Pydantic request and response contracts for the EventTrail HTTP API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

if TYPE_CHECKING:
    from hcmai.event_trail.models import TrailAction, TrailView


class EventTrailOpenRequest(BaseModel):
    """Request payload to initialize an EventTrail session from a KIS search result."""

    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    result_id: str
    expected_kis_revision: int = Field(ge=0)
    search_session_id: str | None = None


class ApproveAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["approve"] = "approve"
    event_id: str


class UseFrameAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["use_frame"] = "use_frame"
    event_id: str
    frame_id: str


class DeclineAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["decline"] = "decline"
    event_id: str


class ClearAnchorAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["clear_anchor"] = "clear_anchor"
    event_id: str


class SetWindowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["set_window"] = "set_window"
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        return self


class ClearWindowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["clear_window"] = "clear_window"


class UndoAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["undo"] = "undo"


EventTrailActionContract = Annotated[
    ApproveAction
    | UseFrameAction
    | DeclineAction
    | ClearAnchorAction
    | SetWindowAction
    | ClearWindowAction
    | UndoAction,
    Field(discriminator="type"),
]


class EventTrailActionRequest(BaseModel):
    """Transactional feedback action applied to an active EventTrail session."""

    model_config = ConfigDict(extra="forbid")

    expected_trail_revision: int = Field(ge=0)
    action: EventTrailActionContract

    def to_domain_action(self) -> TrailAction:
        """Map contract action to its corresponding immutable domain type."""
        from hcmai.event_trail.models import (
            ApproveEvent,
            ClearAnchor,
            ClearWindow,
            DeclineCandidate,
            SetWindow,
            Undo,
            UseFrame,
        )

        action = self.action
        if isinstance(action, ApproveAction):
            return ApproveEvent(event_id=action.event_id)
        elif isinstance(action, UseFrameAction):
            return UseFrame(event_id=action.event_id, frame_id=action.frame_id)
        elif isinstance(action, DeclineAction):
            return DeclineCandidate(event_id=action.event_id)
        elif isinstance(action, ClearAnchorAction):
            return ClearAnchor(event_id=action.event_id)
        elif isinstance(action, SetWindowAction):
            return SetWindow(start_ms=action.start_ms, end_ms=action.end_ms)
        elif isinstance(action, ClearWindowAction):
            return ClearWindow()
        elif isinstance(action, UndoAction):
            return Undo()
        raise ValueError(f"Unsupported action type: {type(action)}")


class EventTrailEventCandidate(BaseModel):
    """One aligned candidate frame coordinate in a trail path."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int


class EventTrailCandidateDiff(BaseModel):
    """Frame/timestamp shift for one event across a revision."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    before_frame_id: str | None = None
    after_frame_id: str | None = None
    before_timestamp_ms: int | None = None
    after_timestamp_ms: int | None = None


class EventTrailTransition(BaseModel):
    """Diff and latency metadata for the most recent committed action."""

    model_config = ConfigDict(extra="forbid")

    action_event_id: str | None = None
    direct_changed_event_ids: list[str] = Field(default_factory=list)
    indirect_changed_event_ids: list[str] = Field(default_factory=list)
    candidate_diffs: list[EventTrailCandidateDiff] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)


class EventTrailSubmissionSelection(BaseModel):
    """Designated frame coordinate for competition submission."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int


class EventTrailStateResponse(BaseModel):
    """Full UI-ready state presentation of an EventTrail session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    result_id: str
    video_id: str
    kis_revision: int
    trail_revision: int
    status: Literal["active", "exhausted"]
    path: list[EventTrailEventCandidate] | None = None
    last_valid_path: list[EventTrailEventCandidate] | None = None
    approved_event_ids: list[str] = Field(default_factory=list)
    rejected_counts: dict[str, int] = Field(default_factory=dict)
    window: tuple[int, int] | None = None
    submission_selection: EventTrailSubmissionSelection | None = None
    transition: EventTrailTransition | None = None

    @classmethod
    def from_domain(cls, view: TrailView) -> EventTrailStateResponse:
        """Construct response from internal TrailView domain projection."""
        path_list = None
        if view.path is not None:
            path_list = [
                EventTrailEventCandidate(
                    event_id=c.event_id,
                    frame_id=c.frame_id,
                    frame_idx=c.frame_idx,
                    timestamp_ms=c.timestamp_ms,
                )
                for c in view.path
            ]

        last_valid_list = None
        if view.last_valid_path is not None:
            last_valid_list = [
                EventTrailEventCandidate(
                    event_id=c.event_id,
                    frame_id=c.frame_id,
                    frame_idx=c.frame_idx,
                    timestamp_ms=c.timestamp_ms,
                )
                for c in view.last_valid_path
            ]

        selection = None
        if view.submission_selection is not None:
            selection = EventTrailSubmissionSelection(
                event_id=view.submission_selection.event_id,
                frame_id=view.submission_selection.frame_id,
                frame_idx=view.submission_selection.frame_idx,
                timestamp_ms=view.submission_selection.timestamp_ms,
            )

        transition = None
        if view.transition is not None:
            transition = EventTrailTransition(
                action_event_id=view.transition.action_event_id,
                direct_changed_event_ids=list(view.transition.direct_changed_event_ids),
                indirect_changed_event_ids=list(view.transition.indirect_changed_event_ids),
                candidate_diffs=[
                    EventTrailCandidateDiff(
                        event_id=d.event_id,
                        before_frame_id=d.before_frame_id,
                        after_frame_id=d.after_frame_id,
                        before_timestamp_ms=d.before_timestamp_ms,
                        after_timestamp_ms=d.after_timestamp_ms,
                    )
                    for d in view.transition.candidate_diffs
                ],
                latency_ms=view.transition.latency_ms,
            )

        return cls(
            session_id=view.session_id,
            result_id=view.result_id,
            video_id=view.video_id,
            kis_revision=view.kis_revision,
            trail_revision=view.trail_revision,
            status=view.status,
            path=path_list,
            last_valid_path=last_valid_list,
            approved_event_ids=list(view.approved_event_ids),
            rejected_counts=view.rejected_counts,
            window=view.window,
            submission_selection=selection,
            transition=transition,
        )
