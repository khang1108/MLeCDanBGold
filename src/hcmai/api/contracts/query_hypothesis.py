"""HTTP contracts for KIS Query Hypothesis API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from hcmai.kis.hypothesis.models import (
    AddEvent,
    AttachImage,
    DetachImage,
    EditEvent,
    MergeEvents,
    QueryHypothesisAction,
    ReorderEvents,
    SplitEvent,
)
from hcmai.kis.models import EventId, KISImageRef, KISIntent, NonBlank


class QueryHypothesisOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: NonBlank
    image_refs: list[KISImageRef] = Field(default_factory=list)


class EditAction(BaseModel):
    type: Literal["edit"] = "edit"
    event_id: EventId
    text: NonBlank


class SplitAction(BaseModel):
    type: Literal["split"] = "split"
    event_id: EventId
    split_at: int = Field(gt=0)
    image_assignments: dict[str, list[Literal["left", "right"]]] = Field(
        default_factory=dict
    )


class MergeAction(BaseModel):
    type: Literal["merge"] = "merge"
    left_event_id: EventId
    right_event_id: EventId


class ReorderAction(BaseModel):
    type: Literal["reorder"] = "reorder"
    event_ids: list[EventId] = Field(min_length=1)


class AddAction(BaseModel):
    type: Literal["add"] = "add"
    position: int = Field(ge=0)
    text: NonBlank | None = None
    images: list[KISImageRef] = Field(default_factory=list)


class AttachImageAction(BaseModel):
    type: Literal["attach_image"] = "attach_image"
    event_id: EventId
    image: KISImageRef


class DetachImageAction(BaseModel):
    type: Literal["detach_image"] = "detach_image"
    event_id: EventId
    asset_id: str


QueryHypothesisActionRequest = Annotated[
    EditAction
    | SplitAction
    | MergeAction
    | ReorderAction
    | AddAction
    | AttachImageAction
    | DetachImageAction,
    Field(discriminator="type"),
]


def to_domain_action(action: QueryHypothesisActionRequest) -> QueryHypothesisAction:
    """Convert an HTTP action payload into its domain equivalent."""
    if isinstance(action, EditAction):
        return EditEvent(event_id=action.event_id, text=action.text)
    if isinstance(action, SplitAction):
        assignments = {
            asset_id: tuple(sides)
            for asset_id, sides in action.image_assignments.items()
        }
        return SplitEvent(
            event_id=action.event_id,
            split_at=action.split_at,
            image_assignments=assignments,
        )
    if isinstance(action, MergeAction):
        return MergeEvents(
            left_event_id=action.left_event_id,
            right_event_id=action.right_event_id,
        )
    if isinstance(action, ReorderAction):
        return ReorderEvents(event_ids=tuple(action.event_ids))
    if isinstance(action, AddAction):
        return AddEvent(
            position=action.position,
            text=action.text,
            images=tuple(action.images),
        )
    if isinstance(action, AttachImageAction):
        return AttachImage(event_id=action.event_id, image=action.image)
    if isinstance(action, DetachImageAction):
        return DetachImage(event_id=action.event_id, asset_id=action.asset_id)
    raise TypeError(f"Unsupported action request: {type(action).__name__}")


class QueryHypothesisMutateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_query_revision: int = Field(ge=0)
    action: QueryHypothesisActionRequest


class QueryHypothesisUndoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_query_revision: int = Field(ge=0)


class QueryHypothesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    intent: KISIntent
    query_revision: int
    can_undo: bool


class QueryHypothesisPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_revision: int
    intent: KISIntent
