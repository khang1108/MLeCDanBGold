"""Transactional process-local progressive-search state."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import time
from typing import Callable, Iterator
from uuid import uuid4

from hcmai.common.schemas import (
    QueryUnit,
    SceneCandidate,
    SearchFilters,
    TaskType,
    TemporalConstraint,
)

from .evidence import ProgressiveEvidenceState


class StaleProgressiveStateError(KeyError):
    """A supplied search ID is absent or expired and must not silently restart."""


class ProgressiveStateConflictError(RuntimeError):
    """A snapshot rewrite or stale concurrent commit cannot be applied safely."""


@dataclass
class ProgressiveSearchState:
    """Committed or proposed state for one progressive KIS search."""

    search_id: str
    task_type: TaskType = TaskType.KIS
    base_filters: SearchFilters | None = None
    session_fingerprint: str | None = None
    version: int = 0
    last_snapshot: str | None = None
    query_units: list[QueryUnit] = field(default_factory=list)
    constraints: list[TemporalConstraint] = field(default_factory=list)
    evidence: ProgressiveEvidenceState = field(default_factory=ProgressiveEvidenceState)
    candidate_video_ids: list[str] = field(default_factory=list)
    ranked_scenes: list[SceneCandidate] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def clone(self) -> ProgressiveSearchState:
        """Return an isolated copy suitable for transactional processing."""

        return deepcopy(self)


class ProgressiveStateStore:
    """Bounded in-memory store with per-search locks and compare-and-swap commits."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        *,
        clock: Callable[[], float] = time,
    ) -> None:
        """Initialize a bounded store with an injectable clock for tests."""

        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("state TTL and max entries must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._states: OrderedDict[str, ProgressiveSearchState] = OrderedDict()
        self._locks: dict[str, RLock] = {}
        self._guard = RLock()

    def new_id(self) -> str:
        """Generate a process-unique opaque progressive search identifier."""

        return f"search-{uuid4().hex}"

    @contextmanager
    def serialized(self, search_id: str) -> Iterator[None]:
        """Serialize all read-process-commit operations for one search ID."""

        with self._guard:
            lock = self._locks.setdefault(search_id, RLock())
        with lock:
            yield

    def get(self, search_id: str) -> ProgressiveSearchState:
        """Return a clone of a live state or reject an unknown/expired ID."""

        with self._guard:
            state = self._states.get(search_id)
            if state is None or self._expired(state):
                if state is not None:
                    self._states.pop(search_id, None)
                    self._locks.pop(search_id, None)
                raise StaleProgressiveStateError(
                    f"progressive search_id {search_id!r} is unknown or expired"
                )
            self._states.move_to_end(search_id)
            return state.clone()

    def create(self, proposed: ProgressiveSearchState) -> ProgressiveSearchState:
        """Commit a successfully processed first request as version one."""

        with self._guard:
            if proposed.search_id in self._states:
                raise ProgressiveStateConflictError("search_id already exists")
            committed = proposed.clone()
            committed.version = 1
            committed.updated_at = self._clock()
            self._states[committed.search_id] = committed
            self._states.move_to_end(committed.search_id)
            self._evict_overflow()
            return committed.clone()

    def commit(
        self,
        proposed: ProgressiveSearchState,
        *,
        expected_version: int,
    ) -> ProgressiveSearchState:
        """Commit a proposed update when its expected version is still current."""

        with self._guard:
            current = self._states.get(proposed.search_id)
            if current is None or self._expired(current):
                raise StaleProgressiveStateError(
                    f"progressive search_id {proposed.search_id!r} is "
                    "unknown or expired"
                )
            if current.version != expected_version:
                raise ProgressiveStateConflictError(
                    "state version changed from "
                    f"v{expected_version} to v{current.version}"
                )
            committed = proposed.clone()
            committed.version = expected_version + 1
            committed.created_at = current.created_at
            committed.updated_at = self._clock()
            self._states[committed.search_id] = committed
            self._states.move_to_end(committed.search_id)
            return committed.clone()

    def cleanup(self) -> int:
        """Remove all expired states and return the number removed."""

        with self._guard:
            expired = [
                key
                for key, state in self._states.items()
                if self._expired(state)
            ]
            for key in expired:
                self._states.pop(key, None)
                self._locks.pop(key, None)
            return len(expired)

    def __len__(self) -> int:
        """Return the current number of committed progressive states."""

        with self._guard:
            return len(self._states)

    def _expired(self, state: ProgressiveSearchState) -> bool:
        """Return whether a state has exceeded its inactivity TTL."""

        return self._clock() - state.updated_at >= self.ttl_seconds

    def _evict_overflow(self) -> None:
        """Evict least-recently-accessed states until the bound is satisfied."""

        while len(self._states) > self.max_entries:
            search_id, _ = self._states.popitem(last=False)
            self._locks.pop(search_id, None)
