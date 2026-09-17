"""Bounded LRU session store, concurrency locking, and undo management for KIS feedback."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Any

from hcmai.api.contracts.feedback import (
    FeedbackStateResponse,
    FeedbackTurnRequest,
    RetrievalOverride,
)
from hcmai.kis.feedback.models import (
    FeedbackCheckpoint,
    FeedbackSession,
)


class FeedbackError(Exception):
    """Domain exception for KIS feedback operations."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(slots=True)
class FeedbackSessionSlot:
    """Internal mutable slot holding a FeedbackSession and its reentrant lock."""

    session: FeedbackSession
    lock: threading.RLock
    in_use: int
    inserted_at: float


class FeedbackSessionStore:
    """Bounded, fixed-TTL store for FeedbackSession instances with per-session locking."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 100,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock if clock is not None else time.monotonic
        self._map_lock = threading.Lock()
        self._entries: OrderedDict[str, FeedbackSessionSlot] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def put(self, session: FeedbackSession) -> None:
        """Store or update a feedback session."""
        now = self._clock()
        with self._map_lock:
            self._tombstones.pop(session.session_id, None)

            if session.session_id in self._entries:
                slot = self._entries[session.session_id]
                slot.session = session
                self._entries.move_to_end(session.session_id)
                return

            if len(self._entries) >= self._max_entries:
                evict_key = None
                for key, slot in self._entries.items():
                    if slot.in_use == 0:
                        evict_key = key
                        break
                if evict_key is None:
                    raise FeedbackError(
                        "FEEDBACK_STORE_BUSY",
                        "Feedback session capacity is busy",
                    )
                self._entries.pop(evict_key)

            new_slot = FeedbackSessionSlot(
                session=session,
                lock=threading.RLock(),
                in_use=0,
                inserted_at=now,
            )
            self._entries[session.session_id] = new_slot

    def get(self, session_id: str) -> FeedbackSession:
        """Read a feedback session, validating TTL and updating LRU order."""
        now = self._clock()
        with self._map_lock:
            if session_id in self._entries:
                slot = self._entries[session_id]
                if now >= slot.inserted_at + self._ttl_seconds:
                    if slot.in_use == 0:
                        self._entries.pop(session_id)
                        self._add_tombstone(session_id)
                    raise FeedbackError(
                        "FEEDBACK_SESSION_EXPIRED",
                        f"Feedback session {session_id} has expired",
                    )
                self._entries.move_to_end(session_id)
                return slot.session

            if session_id in self._tombstones:
                self._tombstones.move_to_end(session_id)
                raise FeedbackError(
                    "FEEDBACK_SESSION_EXPIRED",
                    f"Feedback session {session_id} has expired",
                )

            raise FeedbackError(
                "FEEDBACK_SESSION_NOT_FOUND",
                f"Feedback session {session_id} not found",
            )

    @contextmanager
    def locked(self, session_id: str) -> Iterator[FeedbackSessionSlot]:
        """Acquire the per-session slot lock while tracking in-use status."""
        now = self._clock()
        with self._map_lock:
            if session_id in self._entries:
                slot = self._entries[session_id]
                if now >= slot.inserted_at + self._ttl_seconds:
                    if slot.in_use == 0:
                        self._entries.pop(session_id)
                        self._add_tombstone(session_id)
                    raise FeedbackError(
                        "FEEDBACK_SESSION_EXPIRED",
                        f"Feedback session {session_id} has expired",
                    )
                self._entries.move_to_end(session_id)
                slot.in_use += 1
            elif session_id in self._tombstones:
                self._tombstones.move_to_end(session_id)
                raise FeedbackError(
                    "FEEDBACK_SESSION_EXPIRED",
                    f"Feedback session {session_id} has expired",
                )
            else:
                raise FeedbackError(
                    "FEEDBACK_SESSION_NOT_FOUND",
                    f"Feedback session {session_id} not found",
                )

        try:
            with slot.lock:
                yield slot
        finally:
            with self._map_lock:
                slot.in_use -= 1
                now = self._clock()
                if now >= slot.inserted_at + self._ttl_seconds and slot.in_use == 0:
                    if session_id in self._entries:
                        self._entries.pop(session_id)
                        self._add_tombstone(session_id)

    def remove(self, session_id: str) -> None:
        """Explicitly remove one session without tombstoning."""
        with self._map_lock:
            self._entries.pop(session_id, None)
            self._tombstones.pop(session_id, None)

    def _add_tombstone(self, session_id: str) -> None:
        if len(self._tombstones) >= self._max_entries:
            self._tombstones.popitem(last=False)
        self._tombstones[session_id] = None


def commit_feedback_turn(
    session: FeedbackSession,
    request: FeedbackTurnRequest,
    new_state: FeedbackStateResponse,
    exclusions: tuple[tuple[str, str, str], ...] | None = None,
) -> FeedbackStateResponse:
    """Commit a feedback turn under session lock, verifying revisions and maintaining idempotency."""
    # Check duplicate request_id
    if request.request_id in session.committed_requests:
        cached_payload, cached_response = session.committed_requests[request.request_id]
        if cached_payload == request.model_dump():
            return cached_response
        raise FeedbackError(
            "REQUEST_CONFLICT",
            f"Duplicate request_id {request.request_id} with conflicting payload",
        )

    # Check revisions
    if request.expected_feedback_revision != session.feedback_revision:
        raise FeedbackError(
            "STALE_REVISION",
            f"Expected feedback_revision {request.expected_feedback_revision}, "
            f"current is {session.feedback_revision}",
        )
    if request.expected_kis_revision != session.state.intent.revision:
        raise FeedbackError(
            "STALE_REVISION",
            f"Expected kis_revision {request.expected_kis_revision}, "
            f"current is {session.state.intent.revision}",
        )

    # Snapshot current state into checkpoint
    current = session.state
    checkpoint = FeedbackCheckpoint(
        intent=current.intent,
        retrieval_overrides=dict(current.retrieval_overrides),
        results=list(current.results),
        evidence_snapshot_id=current.evidence_snapshot_id,
        scope=current.scope,
        exclusions=session.exclusions,
        trail_session_id=current.trail.session_id if current.trail else None,
        assistant_message=current.assistant_message,
        changed_event_ids=tuple(current.changed_event_ids),
    )

    # Maintain history bounded to 20 checkpoints
    session.history = (*session.history[-19:], checkpoint)
    if exclusions is not None:
        session.exclusions = exclusions

    final_state = new_state.model_copy(update={"can_undo": len(session.history) > 0})
    session.state = final_state
    session.feedback_revision = final_state.feedback_revision
    session.committed_requests[request.request_id] = (request.model_dump(), final_state)
    return final_state


def apply_prepared_refinement(
    session: FeedbackSession,
    event_id: str,
    text: str,
) -> FeedbackStateResponse:
    """Apply a prepared retrieval refinement without invoking the full retrieval pipeline."""
    current = session.state
    checkpoint = FeedbackCheckpoint(
        intent=current.intent,
        retrieval_overrides=dict(current.retrieval_overrides),
        results=list(current.results),
        evidence_snapshot_id=current.evidence_snapshot_id,
        scope=current.scope,
        exclusions=session.exclusions,
        trail_session_id=current.trail.session_id if current.trail else None,
        assistant_message=current.assistant_message,
        changed_event_ids=tuple(current.changed_event_ids),
    )
    session.history = (*session.history[-19:], checkpoint)

    overrides = dict(current.retrieval_overrides)
    overrides[event_id] = RetrievalOverride(dense_text=text, bm25_text=text)

    next_revision = session.feedback_revision + 1
    updated_state = current.model_copy(
        update={
            "feedback_revision": next_revision,
            "retrieval_overrides": overrides,
            "can_undo": True,
            "changed_event_ids": [event_id],
            "assistant_message": f"Updated retrieval query for {event_id}.",
        }
    )
    session.state = updated_state
    session.feedback_revision = next_revision
    return updated_state


def undo_feedback(session: FeedbackSession) -> FeedbackStateResponse:
    """Restore the previous state checkpoint while issuing a strictly greater feedback revision."""
    if not session.history:
        raise FeedbackError("CANNOT_UNDO", "No feedback checkpoints available to undo")

    last_checkpoint = session.history[-1]
    session.history = session.history[:-1]

    # Revision must increase to reject stale requests
    next_revision = session.feedback_revision + 1
    restored_state = session.state.model_copy(
        update={
            "feedback_revision": next_revision,
            "intent": last_checkpoint.intent,
            "retrieval_overrides": last_checkpoint.retrieval_overrides,
            "results": last_checkpoint.results,
            "evidence_snapshot_id": last_checkpoint.evidence_snapshot_id,
            "scope": last_checkpoint.scope,
            "assistant_message": "Reverted previous feedback change.",
            "changed_event_ids": list(last_checkpoint.changed_event_ids),
            "can_undo": len(session.history) > 0,
        }
    )
    session.exclusions = last_checkpoint.exclusions
    session.state = restored_state
    session.feedback_revision = next_revision
    return restored_state


__all__ = [
    "FeedbackError",
    "FeedbackSessionSlot",
    "FeedbackSessionStore",
    "apply_prepared_refinement",
    "commit_feedback_turn",
    "undo_feedback",
]
