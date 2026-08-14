from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

from hcmai.common.schemas.enum import TaskType
from hcmai.temporal.models import QueryUnit
from hcmai.temporal.query import diff_snapshot
from hcmai.temporal.state import ProgressiveSearchState, ProgressiveStateStore, StateNotFound


@dataclass(frozen=True, slots=True)
class SnapshotPreparation:
    """Proposed state for a KIS/VQA snapshot, pending an explicit commit."""

    state: ProgressiveSearchState
    created: bool
    delta: QueryUnit | None
    warnings: tuple[str, ...]
    commit_required: bool


@dataclass(frozen=True, slots=True)
class StateTaskMismatch(Exception):
    search_id: UUID
    expected_task_type: TaskType
    actual_task_type: TaskType

    def __str__(self) -> str:
        return (
            f"state {self.search_id} is a {self.actual_task_type.value} search, "
            f"cannot prepare a {self.expected_task_type.value} snapshot"
        )


def prepare_snapshot(
    store: ProgressiveStateStore,
    task_type: TaskType,
    snapshot: str,
    *,
    search_id: UUID | None = None,
    allow_missing_state_fallback: bool = False,
) -> SnapshotPreparation:
    warnings: list[str] = []
    if search_id is None:
        current = store.create(task_type)
        created = True
    else:
        stored = store.get(search_id)
        if stored is None:
            if not allow_missing_state_fallback:
                raise StateNotFound(search_id)
            current = store.create(task_type)
            created = True
            warnings.append(f"state {search_id} not found; created a new state")
        else:
            current, created = stored, False
    if current.task_type != task_type:
        raise StateTaskMismatch(current.search_id, task_type, current.task_type)
    diff = diff_snapshot(snapshot, current.last_snapshot, current.query_units)
    warnings.extend(diff.warnings)
    if diff.delta is None:
        return SnapshotPreparation(
            state=current,
            created=created,
            delta=None,
            warnings=tuple(warnings),
            commit_required=False,
        )
    return SnapshotPreparation(
        state=replace(current, last_snapshot=snapshot, query_units=diff.units),
        created=created,
        delta=diff.delta,
        warnings=tuple(warnings),
        commit_required=True,
    )
