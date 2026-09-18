"""Bounded, fixed-TTL store for EventTrail sessions with per-session locking.

This module owns SessionSlot and EventTrailSessionStore for managing active
interactive exploration sessions under capacity constraints and thread-safe locks.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time

from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import EventTrailSession


@dataclass(slots=True)
class SessionSlot:
    """Internal mutable slot holding a session and its reentrant lock."""

    session: EventTrailSession
    lock: threading.RLock
    in_use: int
    inserted_at: float


class EventTrailSessionStore:
    """Bounded, fixed-TTL store for EventTrail sessions with per-session locking."""

    def __init__(
        self,
        ttl_seconds: int,
        max_entries: int,
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
        self._entries: OrderedDict[str, SessionSlot] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def put(self, session: EventTrailSession) -> None:
        """Store or replace a session.

        Under capacity pressure, evicts the oldest LRU slot where in_use == 0.
        If all slots are currently in_use, raises EventTrailError('EVENT_TRAIL_UNAVAILABLE', ...).
        """
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
                    raise EventTrailError(
                        "EVENT_TRAIL_UNAVAILABLE",
                        "EventTrail session capacity is busy",
                    )
                self._entries.pop(evict_key)

            new_slot = SessionSlot(
                session=session,
                lock=threading.RLock(),
                in_use=0,
                inserted_at=now,
            )
            self._entries[session.session_id] = new_slot

    def get(self, session_id: str) -> EventTrailSession:
        """Read a session, checking fixed expiration and updating LRU order."""
        now = self._clock()
        with self._map_lock:
            if session_id in self._entries:
                slot = self._entries[session_id]
                if now >= slot.inserted_at + self._ttl_seconds:
                    if slot.in_use == 0:
                        self._entries.pop(session_id)
                        self._add_tombstone(session_id)
                    raise EventTrailError(
                        "TRAIL_SESSION_EXPIRED",
                        f"Session {session_id} has expired",
                    )
                self._entries.move_to_end(session_id)
                return slot.session

            if session_id in self._tombstones:
                self._tombstones.move_to_end(session_id)
                raise EventTrailError(
                    "TRAIL_SESSION_EXPIRED",
                    f"Session {session_id} has expired",
                )

            raise EventTrailError(
                "TRAIL_SESSION_NOT_FOUND",
                f"Session {session_id} not found",
            )

    @contextmanager
    def locked(self, session_id: str) -> Iterator[SessionSlot]:
        """Acquire the per-session slot lock while tracking in_use."""
        now = self._clock()
        with self._map_lock:
            if session_id in self._entries:
                slot = self._entries[session_id]
                if now >= slot.inserted_at + self._ttl_seconds:
                    if slot.in_use == 0:
                        self._entries.pop(session_id)
                        self._add_tombstone(session_id)
                    raise EventTrailError(
                        "TRAIL_SESSION_EXPIRED",
                        f"Session {session_id} has expired",
                    )
                self._entries.move_to_end(session_id)
                slot.in_use += 1
            elif session_id in self._tombstones:
                self._tombstones.move_to_end(session_id)
                raise EventTrailError(
                    "TRAIL_SESSION_EXPIRED",
                    f"Session {session_id} has expired",
                )
            else:
                raise EventTrailError(
                    "TRAIL_SESSION_NOT_FOUND",
                    f"Session {session_id} not found",
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
