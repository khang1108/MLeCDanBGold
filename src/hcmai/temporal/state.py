from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Generator, MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from time import monotonic
from uuid import UUID, uuid4

from hcmai.common.schemas.enum import TaskType
from hcmai.common.schemas.search import SearchFilters
from hcmai.temporal.models import EvidenceSet, QueryUnit, SceneCandidate, TemporalConstraint


@dataclass(frozen=True, slots=True)
class ProgressiveSearchState:
    search_id: UUID
    task_type: TaskType
    created_at: float
    updated_at: float
    version: int = 0
    last_snapshot: str = ""
    # The filters the stored evidence was gathered under; a change invalidates it.
    filters: SearchFilters | None = None
    query_units: tuple[QueryUnit, ...] = ()
    constraints: tuple[TemporalConstraint, ...] = ()
    evaluated_units_by_video: MutableMapping[str, frozenset[str]] = field(default_factory=dict)
    evidence_by_unit_video: MutableMapping[tuple[str, str], EvidenceSet] = field(
        default_factory=dict
    )
    scene_candidates: tuple[SceneCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class StateNotFound(Exception):
    search_id: UUID

    def __str__(self) -> str:
        return f"state {self.search_id} not found"


@dataclass(frozen=True, slots=True)
class StateVersionConflict(Exception):
    search_id: UUID
    expected_version: int
    actual_version: int

    def __str__(self) -> str:
        return (
            f"state {self.search_id} expected version {self.expected_version} "
            f"but found {self.actual_version}"
        )


@dataclass(slots=True)
class _StoredState:
    """Mutable store entry owned by ProgressiveStateStore."""

    state: ProgressiveSearchState
    lock: RLock


class ProgressiveStateStore:
    def __init__(
        self,
        ttl_seconds: float,
        max_states: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0.0:
            raise InvalidStateStoreConfig("ttl_seconds", ttl_seconds)
        if max_states <= 0:
            raise InvalidStateStoreConfig("max_states", float(max_states))
        self._ttl_seconds = ttl_seconds
        self._max_states = max_states
        self._clock = clock
        self._states: OrderedDict[UUID, _StoredState] = OrderedDict()
        self._guard = RLock()
        self._create_lock = RLock()

    def create(self, task_type: TaskType) -> ProgressiveSearchState:
        now = self._clock()
        state = ProgressiveSearchState(
            search_id=uuid4(),
            task_type=task_type,
            created_at=now,
            updated_at=now,
        )
        with self._create_lock:
            self.cleanup()
            self._evict_until_space()
            with self._guard:
                self._states[state.search_id] = _StoredState(state=state, lock=RLock())
        return deepcopy(state)

    def get(self, search_id: UUID) -> ProgressiveSearchState | None:
        entry = self._entry(search_id)
        if entry is None:
            return None
        with entry.lock:
            now = self._clock()
            with self._guard:
                if self._states.get(search_id) is not entry:
                    return None
                if now - entry.state.updated_at > self._ttl_seconds:
                    del self._states[search_id]
                    return None
                return deepcopy(entry.state)

    def commit(self, state: ProgressiveSearchState, expected_version: int) -> ProgressiveSearchState:
        entry = self._entry_or_raise(state.search_id)
        with entry.lock:
            now = self._clock()
            with self._guard:
                current = self._live_state_or_raise(state.search_id, entry, now)
                if current.version != expected_version:
                    raise StateVersionConflict(state.search_id, expected_version, current.version)
                committed = replace(
                    deepcopy(state),
                    search_id=current.search_id,
                    task_type=current.task_type,
                    created_at=current.created_at,
                    updated_at=now,
                    version=current.version + 1,
                )
                entry.state = committed
                return deepcopy(committed)

    def delete(self, search_id: UUID) -> bool:
        entry = self._entry(search_id)
        if entry is None:
            return False
        with entry.lock:
            with self._guard:
                if self._states.get(search_id) is not entry:
                    return False
                del self._states[search_id]
                return True

    def cleanup(self) -> int:
        removed = 0
        with self._guard:
            search_ids = tuple(self._states)
        for search_id in search_ids:
            entry = self._entry(search_id)
            if entry is None:
                continue
            with entry.lock:
                now = self._clock()
                with self._guard:
                    if self._states.get(search_id) is not entry:
                        continue
                    if now - entry.state.updated_at > self._ttl_seconds:
                        del self._states[search_id]
                        removed += 1
        return removed

    @contextmanager
    def lock(self, search_id: UUID) -> Generator[None]:
        entry = self._entry_or_raise(search_id)
        with entry.lock:
            with self._guard:
                self._live_state_or_raise(search_id, entry, self._clock())
            yield

    def _evict_until_space(self) -> None:
        """Drop the oldest states until a new one fits, waiting for their holders."""
        while True:
            with self._guard:
                if len(self._states) < self._max_states:
                    return
                search_id, entry = next(iter(self._states.items()))
            with entry.lock:
                with self._guard:
                    if self._states.get(search_id) is entry:
                        del self._states[search_id]

    def _entry(self, search_id: UUID) -> _StoredState | None:
        with self._guard:
            return self._states.get(search_id)

    def _entry_or_raise(self, search_id: UUID) -> _StoredState:
        entry = self._entry(search_id)
        if entry is None:
            raise StateNotFound(search_id)
        return entry

    def _live_state_or_raise(
        self,
        search_id: UUID,
        entry: _StoredState,
        now: float,
    ) -> ProgressiveSearchState:
        if self._states.get(search_id) is not entry:
            raise StateNotFound(search_id)
        if now - entry.state.updated_at > self._ttl_seconds:
            del self._states[search_id]
            raise StateNotFound(search_id)
        return entry.state


@dataclass(frozen=True, slots=True)
class InvalidStateStoreConfig(ValueError):
    field_name: str
    value: float

    def __str__(self) -> str:
        return f"{self.field_name} must be positive, got {self.value}"
