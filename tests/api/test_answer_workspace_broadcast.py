"""Workspace task-scope notification isolation tests."""

from __future__ import annotations

import asyncio

from hcmai.api.contracts.workspace import AnswerWorkspaceEvent, AnswerWorkspaceSnapshot
from hcmai.api.routers.workspace import WorkspaceSocketHub


class _Socket:
    """Capture JSON events sent by the in-process workspace hub."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


def _snapshot(evaluation_id: str, task_scope_key: str) -> AnswerWorkspaceSnapshot:
    """Build one empty durable scope for an outbound event."""

    return AnswerWorkspaceSnapshot(
        avs_enabled=False,
        revision=1,
        updated_by_user_id="member-a",
        updated_at_ms=1,
        active_evaluation_id=evaluation_id,
        active_task_scope_key=task_scope_key,
        active_task_name=f"Task {task_scope_key}",
    )


def test_workspace_broadcast_is_limited_to_matching_evaluation_and_task() -> None:
    """A team event for one live task must not reach another task's socket."""

    hub = WorkspaceSocketHub()
    task_a = _Socket()
    task_b = _Socket()
    hub.connect(task_a, "eval-a", "task-a")
    hub.connect(task_b, "eval-b", "task-b")
    event = AnswerWorkspaceEvent(
        type="answer.added",
        workspace=_snapshot("eval-a", "task-a"),
    )

    asyncio.run(hub.broadcast(event))

    assert len(task_a.events) == 1
    assert task_a.events[0]["workspace"]["active_task_scope_key"] == "task-a"
    assert task_b.events == []


def test_task_switch_rebinds_old_and_new_collaborators_only() -> None:
    """Deliver the committed switch to both sides without crossing other scopes."""

    hub = WorkspaceSocketHub()
    old_collaborator = _Socket()
    new_collaborator = _Socket()
    unrelated = _Socket()
    hub.connect(old_collaborator, "eval-old", "task-old")
    hub.connect(new_collaborator, "eval-new", "task-new")
    hub.connect(unrelated, "eval-other", "task-other")
    event = AnswerWorkspaceEvent(
        type="answer.task.switched",
        workspace=_snapshot("eval-new", "task-new"),
    )

    asyncio.run(hub.rebind_and_broadcast_task_switch(
        event,
        old_evaluation_id="eval-old",
        old_task_scope_key="task-old",
    ))

    expected_payload = event.model_dump(mode="json")
    assert old_collaborator.events == [expected_payload]
    assert new_collaborator.events == [expected_payload]
    assert unrelated.events == []
    assert hub.scope_for(old_collaborator) == ("eval-new", "task-new")
    assert hub.scope_for(new_collaborator) == ("eval-new", "task-new")
    assert hub.scope_for(unrelated) == ("eval-other", "task-other")
