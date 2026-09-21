"""Domain models and action definitions for KIS Query Hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hcmai.kis.models import KISImageRef, KISIntent


class QueryHypothesisError(Exception):
    """Raised when a query hypothesis operation fails validation or encounters a conflict."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class SplitEvent:
    """Split an existing event at a character offset with explicit image assignments."""

    event_id: str
    split_at: int
    image_assignments: dict[str, tuple[Literal["left", "right"], ...]]


@dataclass(frozen=True, slots=True)
class MergeEvents:
    """Merge two adjacent events into one."""

    left_event_id: str
    right_event_id: str


@dataclass(frozen=True, slots=True)
class ReorderEvents:
    """Reorder existing events by their IDs."""

    event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EditEvent:
    """Directly override canonical text for an event."""

    event_id: str
    text: str


@dataclass(frozen=True, slots=True)
class AddEvent:
    """Add a new user-authored event at an index position."""

    position: int
    text: str | None
    images: tuple[KISImageRef, ...] = ()


@dataclass(frozen=True, slots=True)
class AttachImage:
    """Attach an image reference to an existing event."""

    event_id: str
    image: KISImageRef


@dataclass(frozen=True, slots=True)
class DetachImage:
    """Detach an image reference from an existing event."""

    event_id: str
    asset_id: str


QueryHypothesisAction = (
    SplitEvent
    | MergeEvents
    | ReorderEvents
    | EditEvent
    | AddEvent
    | AttachImage
    | DetachImage
)


@dataclass(frozen=True, slots=True)
class QueryHypothesisCheckpoint:
    """Historical checkpoint for undoing query mutations."""

    intent: KISIntent


@dataclass(slots=True)
class QueryHypothesisSession:
    """Bounded session owning canonical query hypothesis state and mutation history."""

    session_id: str
    intent: KISIntent
    history: tuple[QueryHypothesisCheckpoint, ...]
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class QueryHypothesisView:
    """Client-facing presentation view of a Query Hypothesis session."""

    session_id: str
    intent: KISIntent
    query_revision: int
    can_undo: bool

    @classmethod
    def from_session(cls, session: QueryHypothesisSession) -> QueryHypothesisView:
        return cls(
            session_id=session.session_id,
            intent=session.intent,
            query_revision=session.intent.revision,
            can_undo=bool(session.history),
        )


@dataclass(frozen=True, slots=True)
class QueryHypothesisPreview:
    """Preview of a proposed query action without mutating session state."""

    base_revision: int
    intent: KISIntent
