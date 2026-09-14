"""Focused transaction tests for immutable DRES submission reservations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmai.api.history import (
    AnswerSubmissionInFlight,
    AnswerWorkspaceConflict,
    WorkspaceStore,
)


def _frame(store: WorkspaceStore, *, at: int, revision: int | None = None) -> str:
    """Add one canonical point candidate and return its stable ID."""

    if revision is None:
        revision = store.get_answer_workspace("eval-a", "scope-a", "KIS task").revision
    created = store.add_frame_candidate(
        video_id="video-a",
        timestamp_ms=at,
        source_frame_id=f"frame-{at}",
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
        expected_workspace_revision=revision,
    )
    return created.candidate.candidate_id


def _payload(video: str, at: int) -> dict[str, object]:
    """Return one exact JSON-ready DRES answer body."""

    return {
        "answerSets": [{
            "taskName": "KIS task",
            "answers": [{
                "mediaItemName": video,
                "start": at,
                "end": at,
            }],
        }],
    }


def test_reservation_persists_exact_payload_and_freezes_workspace(tmp_path: Path) -> None:
    """Reserve candidate membership and wire JSON before any network call."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    candidate_id = _frame(store, at=12_346)
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    payload = _payload("video-a", 12_346)

    attempt = store.reserve_submission(
        kind="KIS",
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
        expected_workspace_revision=current.revision,
        candidates=[{"candidate_id": candidate_id, "expected_revision": 1}],
        dres_payload=payload,
    )

    snapshot = json.loads(attempt.snapshot_json)
    assert attempt.task_name == "KIS task"
    assert snapshot["dres_payload"] == payload
    assert snapshot["candidate_revisions"] == [{
        "candidate_id": candidate_id,
        "revision": 1,
    }]
    assert attempt.state == "FORWARDING"
    with pytest.raises(AnswerSubmissionInFlight):
        store.add_frame_candidate(
            video_id="video-b",
            timestamp_ms=22_000,
            source_frame_id=None,
            user_id="member-b",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
            expected_workspace_revision=attempt.workspace_revision,
        )


@pytest.mark.parametrize(
    "answer_set",
    [
        {"taskId": "scope-a", "answers": [{"mediaItemName": "video-a", "start": 12_346, "end": 12_346}]},
        {"taskName": "Different task", "answers": [{"mediaItemName": "video-a", "start": 12_346, "end": 12_346}]},
    ],
)
def test_reservation_requires_matching_task_name_and_rejects_task_id(
    tmp_path: Path,
    answer_set: dict[str, object],
) -> None:
    """Keep internal scope keys private and bind the frozen task name exactly."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    candidate_id = _frame(store, at=12_346)
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    payload = {"answerSets": [answer_set]}

    with pytest.raises(AnswerWorkspaceConflict, match="task name"):
        store.reserve_submission(
            kind="KIS",
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
            task_name="KIS task",
            expected_workspace_revision=current.revision,
            candidates=[{"candidate_id": candidate_id, "expected_revision": 1}],
            dres_payload=payload,
        )

    assert store.get_answer_workspace("eval-a", "scope-a", "KIS task").pending_submission is None


@pytest.mark.parametrize(
    "kind,candidate_kind,avs_enabled",
    [
        ("KIS", "TEXT", False),
        ("VQA", "FRAME", False),
        ("KIS", "FRAME", True),
        ("VQA", "TEXT", True),
    ],
)
def test_single_candidate_reservation_checks_mode_and_answer_kind(
    tmp_path: Path,
    kind: str,
    candidate_kind: str,
    avs_enabled: bool,
) -> None:
    """Never send the wrong candidate kind or stale task mode as KIS/VQA."""

    store = WorkspaceStore(tmp_path / f"{kind}-{candidate_kind}-{avs_enabled}.sqlite3")
    snapshot = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    if avs_enabled:
        mode = store.set_avs_enabled(
            avs_enabled=True,
            expected_workspace_revision=snapshot.revision,
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
        )
        revision = mode.workspace_revision
    else:
        revision = snapshot.revision
    if candidate_kind == "FRAME":
        candidate_id = _frame(store, at=1_000, revision=revision)
        payload = _payload("video-a", 1_000)
    else:
        candidate = store.add_text_candidate(
            text="spoken answer",
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
            expected_workspace_revision=revision,
        )
        candidate_id = candidate.candidate.candidate_id
        payload = {"answerSets": [{"taskName": "KIS task", "answers": [{"text": "spoken answer"}]}]}
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")

    with pytest.raises(AnswerWorkspaceConflict):
        store.reserve_submission(
            kind=kind,
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
            expected_workspace_revision=current.revision,
            candidates=[{"candidate_id": candidate_id, "expected_revision": 1}],
            dres_payload=payload,
        )


def test_avs_reservation_requires_every_unsubmitted_frame_in_workspace_order(
    tmp_path: Path,
) -> None:
    """Freeze one non-empty, complete, ordered AVS candidate set."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    empty = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    mode = store.set_avs_enabled(
        avs_enabled=True,
        expected_workspace_revision=empty.revision,
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
    )
    first = _frame(store, at=1_000, revision=mode.workspace_revision)
    after_first = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    second = _frame(store, at=2_000, revision=after_first.revision)
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    candidates = [
        {"candidate_id": item.candidate_id, "expected_revision": item.revision}
        for item in current.candidates
    ]
    payload = {
        "answerSets": [{
            "taskName": "KIS task",
            "answers": [
                {"mediaItemName": "video-a", "start": 1_000, "end": 1_000},
                {"mediaItemName": "video-a", "start": 2_000, "end": 2_000},
            ],
        }],
    }

    with pytest.raises(AnswerWorkspaceConflict):
        store.reserve_submission(
            kind="AVS",
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
            expected_workspace_revision=current.revision,
            candidates=list(reversed(candidates)),
            dres_payload=payload,
        )
    with pytest.raises(AnswerWorkspaceConflict):
        store.reserve_submission(
            kind="AVS",
            user_id="member-a",
            evaluation_id="eval-a",
            task_scope_key="scope-a",
        task_name="KIS task",
            expected_workspace_revision=current.revision,
            candidates=candidates[:1],
            dres_payload=payload,
        )

    reservation = store.reserve_submission(
        kind="AVS",
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
        expected_workspace_revision=current.revision,
        candidates=candidates,
        dres_payload=payload,
    )
    frozen = json.loads(reservation.snapshot_json)
    assert [item["candidate_id"] for item in frozen["candidate_revisions"]] == [first, second]
    assert len(frozen["dres_payload"]["answerSets"][0]["answers"]) == 2


def test_restart_changes_forwarding_to_unknown_and_explicit_resolution_is_exact(
    tmp_path: Path,
) -> None:
    """Recover ambiguous sends as locked UNKNOWN attempts, never auto-retry."""

    database = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(database)
    candidate_id = _frame(store, at=12_346)
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    attempt = store.reserve_submission(
        kind="KIS",
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
        expected_workspace_revision=current.revision,
        candidates=[{"candidate_id": candidate_id, "expected_revision": 1}],
        dres_payload=_payload("video-a", 12_346),
    )

    restarted = WorkspaceStore(database)
    pending = restarted.get_answer_workspace("eval-a", "scope-a", "KIS task")
    assert pending.pending_submission is not None
    assert pending.pending_submission.state == "UNKNOWN"

    accepted = restarted.resolve_unknown_submission(attempt.attempt_id, accepted=True)
    assert accepted.pending_submission is None
    submitted = next(item for item in accepted.candidates if item.candidate_id == candidate_id)
    assert submitted.submitted_at_ms is not None
    assert submitted.dres_status == "accepted"
    assert restarted.get_submission_attempt(attempt.attempt_id).state == "ACCEPTED"


def test_definitive_rejection_releases_attempt_without_marking_candidate(
    tmp_path: Path,
) -> None:
    """Allow an explicit later retry after a definitive DRES rejection."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    candidate_id = _frame(store, at=12_346)
    current = store.get_answer_workspace("eval-a", "scope-a", "KIS task")
    attempt = store.reserve_submission(
        kind="KIS",
        user_id="member-a",
        evaluation_id="eval-a",
        task_scope_key="scope-a",
        task_name="KIS task",
        expected_workspace_revision=current.revision,
        candidates=[{"candidate_id": candidate_id, "expected_revision": 1}],
        dres_payload=_payload("video-a", 12_346),
    )

    released = store.complete_submission(
        attempt.attempt_id,
        accepted=False,
        dres_status="rejected",
    )
    candidate = released.candidates[0]
    assert candidate.submitted_at_ms is None
    assert released.pending_submission is None
    assert store.get_submission_attempt(attempt.attempt_id).state == "NOT_ACCEPTED"
