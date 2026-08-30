"""Task-agnostic contracts for ordered event-to-frame alignment.

This module owns the shared plan and canonical output path used by temporal
alignment. It does not select a retrieval model, run dynamic programming, or
shape KIS and TRAKE HTTP responses.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString
from .frame import FrameRecord
from .search import SearchFilters


class AlignmentEvent(ContractModel):
    """One ordered semantic event supplied to task-agnostic alignment."""

    event_id: NonEmptyString
    text: NonEmptyString
    order: int = Field(ge=0)


class AlignmentPlan(ContractModel):
    """Validated ordered events and optional retrieval restrictions.

    Event order is explicit so callers cannot accidentally use array position
    as a replacement for the semantic order supplied by a query planner.
    """

    events: tuple[AlignmentEvent, ...] = Field(min_length=1)
    filters: SearchFilters | None = None

    @model_validator(mode="after")
    def validate_event_order(self) -> Self:
        """Require unique, zero-based consecutive event identities and order."""

        orders = [event.order for event in self.events]
        if orders != list(range(len(self.events))):
            raise ValueError("alignment event order must be consecutive")

        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("alignment event IDs must be unique")
        return self


class AlignmentPath(ContractModel):
    """One same-video canonical frame path aligned to ordered event IDs.

    ``frames`` retain complete canonical frame metadata. ``event_ids`` map each
    frame position back to the plan without allowing a task head to invent or
    rewrite frame identity.
    """

    path_id: NonEmptyString
    video_id: NonEmptyString
    frames: tuple[FrameRecord, ...] = Field(min_length=1)
    event_ids: tuple[NonEmptyString, ...] = Field(min_length=1)
    score: float

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Keep path/event cardinality, identity, and chronology traceable."""

        if len(self.frames) != len(self.event_ids):
            raise ValueError("alignment path must contain one frame per event")
        if len(self.event_ids) != len(set(self.event_ids)):
            raise ValueError("alignment path event IDs must be unique")
        if any(frame.video_id != self.video_id for frame in self.frames):
            raise ValueError("alignment path frames must share video_id")
        if any(
            current.timestamp_ms < previous.timestamp_ms
            for previous, current in zip(self.frames, self.frames[1:])
        ):
            raise ValueError("alignment path frames must be chronological")
        return self
