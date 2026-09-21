"""Bounded in-memory TTL session store for KIS Query Hypotheses with locking."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterator

from hcmai.kis.hypothesis.models import (
    QueryHypothesisError,
    QueryHypothesisSession,
)


@dataclass(slots=True)
class QueryHypothesisSlot:
    """Internal storage slot managing per-session lock, concurrency, and TTL."""

    session: QueryHypothesisSession
    lock: threading.RLock
    in_use: int
    inserted_at: float


class QueryHypothesisStore:
    """Thread-safe bounded in-memory store for Query Hypothesis sessions with TTL."""

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_entries: int = 100,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock if clock is not None else time.monotonic
        self._map_lock = threading.Lock()
        self._entries: OrderedDict[str, QueryHypothesisSlot] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def put(self, session: QueryHypothesisSession) -> None:
        """Store or update a session slot, evicting oldest idle entry if capacity is exceeded."""
        now = self._clock()
        with self._map_lock:
            self._tombstones.pop(session.session_id, None)
            if session.session_id in self._entries:
                slot = self._entries[session.session_id]
                slot.session = session
                self._entries.move_to_end(session.session_id)
                return
            if len(self._entries) >= self._max_entries:
                evict_key = next(
                    (key for key, slot in self._entries.items() if slot.in_use == 0),
                    None,
                )
                if evict_key is None:
                    raise QueryHypothesisError(
                        "QUERY_HYPOTHESIS_UNAVAILABLE",
                        "Query hypothesis capacity is busy",
                    )
                self._entries.pop(evict_key)
            self._entries[session.session_id] = QueryHypothesisSlot(
                session=session,
                lock=threading.RLock(),
                in_use=0,
                inserted_at=now,
            )

    def get(self, session_id: str) -> QueryHypothesisSession:
        """Retrieve a session by ID under lock, validating TTL."""
        with self.locked(session_id) as slot:
            return slot.session

    @contextmanager
    def locked(self, session_id: str) -> Iterator[QueryHypothesisSlot]:
        """Context manager acquiring the per-session lock with in_use tracking and TTL check."""
        now = self._clock()
        with self._map_lock:
            slot = self._entries.get(session_id)
            if slot is None:
                code = (
                    "QUERY_HYPOTHESIS_EXPIRED"
                    if session_id in self._tombstones
                    else "QUERY_HYPOTHESIS_NOT_FOUND"
                )
                raise QueryHypothesisError(
                    code, f"Query hypothesis {session_id} is unavailable"
                )
            if now >= slot.inserted_at + self._ttl_seconds:
                if slot.in_use == 0:
                    self._entries.pop(session_id, None)
                    self._tombstones[session_id] = None
                raise QueryHypothesisError(
                    "QUERY_HYPOTHESIS_EXPIRED",
                    f"Query hypothesis {session_id} has expired",
                )
            self._entries.move_to_end(session_id)
            slot.in_use += 1
        try:
            with slot.lock:
                yield slot
        finally:
            with self._map_lock:
                slot.in_use -= 1

    def remove(self, session_id: str) -> None:
        """Remove a session by ID and record a tombstone."""
        with self._map_lock:
            self._entries.pop(session_id, None)
            self._tombstones.pop(session_id, None)
