"""Transactional SQLite tests for the structured collaborative answer workspace."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hcmai.api.history import (
    AnswerWorkspaceConflict,
    AnswerWorkspaceTaskScopeMismatch,
    WorkspaceStore,
)


def _legacy_database(path: Path) -> None:
    """Create the schema that predates the versioned answer-workspace migration."""

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE query_history (
          query_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          query_text TEXT NOT NULL,
          result_snapshot_json TEXT NOT NULL,
          submission_file_names_json TEXT NOT NULL DEFAULT '[]',
          viewed_frame_ids_json TEXT NOT NULL DEFAULT '[]',
          submitted_frame_ids_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL
        );
        CREATE INDEX query_history_user_created_idx
          ON query_history (user_id, created_at DESC);
        CREATE TABLE submission_files (
          name TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          is_validated INTEGER NOT NULL DEFAULT 0,
          revision INTEGER NOT NULL
        );
        INSERT INTO query_history VALUES
          ('q1', 'member-1', 'query', '{}', '["old.csv"]', '["frame-a"]', '["frame-a"]', '2026-01-01T00:00:00Z');
        INSERT INTO submission_files VALUES ('old.csv', 'old data', 1, 3);
        """
    )
    connection.commit()
    connection.close()


def _version_one_workspace_database(path: Path, *, populated: bool) -> str | None:
    """Create the exact task-ID schema shipped by answer-workspace database v1."""

    snapshot_json = (
        '{ "kind":"KIS", "evaluation_id":"eval-old", "task_id":"task-old", '
        '"submitted_by_user_id":"member-old", "workspace_revision":12, '
        '"candidate_revisions":[{"candidate_id":"frame-submitting", "revision":3}], '
        '"dres_payload":{"answerSets":[{"taskId":"task-old","answers":[]}]}} '
        if populated
        else None
    )
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE answer_workspace_state (
          singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
          avs_enabled INTEGER NOT NULL CHECK (avs_enabled IN (0, 1)),
          evaluation_id TEXT,
          task_id TEXT,
          revision INTEGER NOT NULL CHECK (revision >= 0),
          pending_submission_id TEXT,
          pending_submission_state TEXT CHECK (
            pending_submission_state IS NULL
            OR pending_submission_state IN ('FORWARDING', 'UNKNOWN')
          ),
          updated_by_user_id TEXT NOT NULL,
          updated_at_ms INTEGER NOT NULL,
          CHECK ((evaluation_id IS NULL) = (task_id IS NULL)),
          CHECK ((pending_submission_id IS NULL) = (pending_submission_state IS NULL))
        );
        CREATE TABLE answer_candidates (
          candidate_id TEXT PRIMARY KEY,
          kind TEXT NOT NULL CHECK (kind IN ('FRAME', 'TEXT')),
          source_frame_id TEXT,
          video_id TEXT,
          timestamp_ms INTEGER,
          text TEXT,
          contributed_by_user_id TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL,
          revision INTEGER NOT NULL CHECK (revision > 0),
          submitted_at_ms INTEGER,
          submitted_by_user_id TEXT,
          dres_status TEXT,
          evaluation_id TEXT NOT NULL,
          task_id TEXT NOT NULL,
          CHECK (
            (kind = 'FRAME' AND video_id IS NOT NULL AND timestamp_ms IS NOT NULL AND text IS NULL)
            OR
            (kind = 'TEXT' AND text IS NOT NULL AND source_frame_id IS NULL
              AND video_id IS NULL AND timestamp_ms IS NULL)
          )
        );
        CREATE UNIQUE INDEX answer_frame_unique
          ON answer_candidates(video_id, timestamp_ms) WHERE kind = 'FRAME';
        CREATE TABLE submission_attempts (
          attempt_id TEXT PRIMARY KEY,
          kind TEXT NOT NULL CHECK (kind IN ('KIS', 'VQA', 'AVS')),
          evaluation_id TEXT NOT NULL,
          task_id TEXT NOT NULL,
          submitted_by_user_id TEXT NOT NULL,
          workspace_revision INTEGER NOT NULL,
          snapshot_json TEXT NOT NULL,
          state TEXT NOT NULL CHECK (
            state IN ('FORWARDING', 'UNKNOWN', 'ACCEPTED', 'NOT_ACCEPTED')
          ),
          created_at_ms INTEGER NOT NULL,
          resolved_at_ms INTEGER
        );
        PRAGMA user_version = 1;
        """
    )
    if populated:
        assert snapshot_json is not None
        connection.execute(
            """INSERT INTO answer_workspace_state VALUES
            (1, 1, 'eval-old', 'task-old', 12, 'attempt-old', 'FORWARDING',
             'member-old', 1700000000000)"""
        )
        connection.executemany(
            """INSERT INTO answer_candidates VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    "frame-submitting", "FRAME", "source-a", "video-a", 12345,
                    None, "member-old", 100, 3, None, None, None, "eval-old", "task-old",
                ),
                (
                    "frame-submitted", "FRAME", "source-b", "video-b", 45678,
                    None, "member-two", 200, 2, 1699999999999, "member-two", "accepted",
                    "eval-old", "task-old",
                ),
                (
                    "text-old", "TEXT", None, None, None, "answer", "member-old", 300,
                    4, None, None, None, "eval-old", "task-old",
                ),
            ],
        )
        connection.execute(
            """INSERT INTO submission_attempts VALUES
            ('attempt-old', 'KIS', 'eval-old', 'task-old', 'member-old', 12, ?,
             'FORWARDING', 1700000000000, NULL)""",
            (snapshot_json,),
        )
    connection.commit()
    connection.close()
    return snapshot_json


def test_migration_is_versioned_and_retires_only_submission_file_history(tmp_path: Path) -> None:
    """Preserve replay/view history while dropping the retired CSV data model."""

    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)

    WorkspaceStore(path)

    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    history_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(query_history)")
    }
    viewed, = connection.execute(
        "SELECT viewed_frame_ids_json FROM query_history WHERE query_id='q1'"
    ).fetchone()
    connection.close()

    assert version == 2
    assert "submission_files" not in tables
    assert {"answer_workspace_state", "answer_candidates", "submission_attempts"} <= tables
    assert "submission_file_names_json" not in history_columns
    assert "submitted_frame_ids_json" not in history_columns
    assert viewed == '["frame-a"]'


def test_failed_schema_step_rolls_back_legacy_migration_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback earlier DDL and data changes when a later migration step fails."""

    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)
    create_tables = WorkspaceStore._create_answer_tables

    def fail_after_creating_tables(self, connection) -> None:
        create_tables(connection)
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(WorkspaceStore, "_create_answer_tables", fail_after_creating_tables)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        WorkspaceStore(path)

    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    submission_names, viewed = connection.execute(
        "SELECT submission_file_names_json, viewed_frame_ids_json "
        "FROM query_history WHERE query_id='q1'"
    ).fetchone()
    connection.close()

    assert version == 0
    assert "submission_files" in tables
    assert "answer_workspace_state" not in tables
    assert submission_names == '["old.csv"]'
    assert viewed == '["frame-a"]'


def test_idle_version_one_workspace_migrates_to_task_scope_schema(tmp_path: Path) -> None:
    """Rebuild empty v1 answer tables and install the unique frame index."""

    path = tmp_path / "idle-v1.sqlite3"
    _version_one_workspace_database(path, populated=False)

    WorkspaceStore(path)

    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    columns = {
        table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        for table in ("answer_workspace_state", "answer_candidates", "submission_attempts")
    }
    connection.close()

    assert version == 2
    assert {"task_scope_key", "task_name"} <= columns["answer_workspace_state"]
    assert "task_id" not in columns["answer_workspace_state"]
    assert "task_scope_key" in columns["answer_candidates"]
    assert "task_id" not in columns["answer_candidates"]
    assert {"task_scope_key", "task_name"} <= columns["submission_attempts"]
    assert "task_id" not in columns["submission_attempts"]


def test_populated_version_one_workspace_migrates_and_recovers_attempt(tmp_path: Path) -> None:
    """Preserve v1 audit bytes and candidate state while recovering a send."""

    path = tmp_path / "populated-v1.sqlite3"
    snapshot_json = _version_one_workspace_database(path, populated=True)

    store = WorkspaceStore(path)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    state = connection.execute("SELECT * FROM answer_workspace_state").fetchone()
    candidates = connection.execute(
        "SELECT * FROM answer_candidates ORDER BY created_at_ms, candidate_id"
    ).fetchall()
    attempt = connection.execute("SELECT * FROM submission_attempts").fetchone()
    indexes = connection.execute("PRAGMA index_list(answer_candidates)").fetchall()
    connection.close()

    assert state["task_scope_key"] == "task-old"
    assert state["task_name"] == "legacy-unverified"
    assert state["pending_submission_id"] == "attempt-old"
    assert state["pending_submission_state"] == "UNKNOWN"
    assert [(row["candidate_id"], row["revision"], row["submitted_at_ms"]) for row in candidates] == [
        ("frame-submitting", 3, None),
        ("frame-submitted", 2, 1699999999999),
        ("text-old", 4, None),
    ]
    assert all(row["task_scope_key"] == "task-old" for row in candidates)
    assert attempt["task_scope_key"] == "task-old"
    assert attempt["task_name"] == "legacy-unverified"
    assert attempt["state"] == "UNKNOWN"
    assert attempt["snapshot_json"] == snapshot_json
    assert any(row["name"] == "answer_frame_unique" and row["unique"] for row in indexes)

    with sqlite3.connect(path) as duplicate_connection:
        with pytest.raises(sqlite3.IntegrityError):
            duplicate_connection.execute(
                """INSERT INTO answer_candidates (
                  candidate_id, kind, video_id, timestamp_ms, contributed_by_user_id,
                  created_at_ms, revision, evaluation_id, task_scope_key
                ) VALUES ('duplicate-frame', 'FRAME', 'video-a', 12345, 'member',
                          400, 1, 'eval-old', 'task-old')"""
            )

    resolved = store.resolve_unknown_submission("attempt-old", accepted=True)
    assert resolved.pending_submission is None
    submitted = next(item for item in resolved.candidates if item.candidate_id == "frame-submitting")
    assert submitted.submitted_at_ms is not None
    assert submitted.dres_status == "accepted"
    preserved_attempt = store.get_submission_attempt("attempt-old")
    assert preserved_attempt.state == "ACCEPTED"
    assert preserved_attempt.snapshot_json == snapshot_json

    active = store.get_answer_workspace(
        "eval-current",
        "dres-task-v1:current-scope",
        "Current KIS task",
    )
    assert active.task_scope_mismatch is True
    assert active.task_name == "legacy-unverified"
    assert active.active_task_name == "Current KIS task"
    with pytest.raises(AnswerWorkspaceTaskScopeMismatch):
        store.add_frame_candidate(
            video_id="video-current",
            timestamp_ms=7,
            source_frame_id=None,
            user_id="member-new",
            evaluation_id="eval-current",
            task_scope_key="dres-task-v1:current-scope",
            task_name="Current KIS task",
            expected_workspace_revision=active.revision,
        )

    switched = store.clear_and_switch_task(
        expected_workspace_revision=active.revision,
        expected_old_evaluation_id="eval-old",
        expected_old_task_scope_key="task-old",
        target_evaluation_id="eval-current",
        target_task_scope_key="dres-task-v1:current-scope",
        target_task_name="Current KIS task",
        user_id="member-new",
    )
    assert switched.task_scope_mismatch is False
    assert switched.task_scope_key == "dres-task-v1:current-scope"
    assert switched.task_name == "Current KIS task"
    assert switched.candidates == []


def test_failed_v1_rebuild_rolls_back_schema_data_and_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep v1 untouched when any step in the table rebuild fails."""

    path = tmp_path / "rollback-v1.sqlite3"
    snapshot_json = _version_one_workspace_database(path, populated=True)
    create_tables = WorkspaceStore._create_answer_tables

    def fail_after_creating_tables(connection) -> None:
        create_tables(connection)
        raise RuntimeError("injected v1 migration failure")

    monkeypatch.setattr(WorkspaceStore, "_create_answer_tables", fail_after_creating_tables)
    with pytest.raises(RuntimeError, match="injected v1 migration failure"):
        WorkspaceStore(path)

    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    state = connection.execute(
        "SELECT task_id, pending_submission_id, pending_submission_state "
        "FROM answer_workspace_state"
    ).fetchone()
    attempt = connection.execute(
        "SELECT task_id, snapshot_json, state FROM submission_attempts"
    ).fetchone()
    index_names = {
        row[1] for row in connection.execute("PRAGMA index_list(answer_candidates)")
    }
    connection.close()

    assert version == 1
    assert state == ("task-old", "attempt-old", "FORWARDING")
    assert attempt == ("task-old", snapshot_json, "FORWARDING")
    assert "answer_frame_unique" in index_names


def test_fresh_store_creates_versioned_workspace_and_empty_snapshot(tmp_path: Path) -> None:
    """Create persistent workspace state on an empty SQLite database."""

    store = WorkspaceStore(tmp_path / "fresh.sqlite3")

    workspace = store.get_answer_workspace("eval-1", "task-1", "KIS task")

    assert workspace.revision >= 1
    assert workspace.evaluation_id == "eval-1"
    assert workspace.task_scope_key == "task-1"
    assert workspace.candidates == []
    assert workspace.avs_enabled is False


def test_empty_workspace_automatically_rebinds_and_v1_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    """Allow empty scopes to follow a task transition without losing mode state."""

    path = tmp_path / "workspace.sqlite3"
    store = WorkspaceStore(path)
    initial = store.get_answer_workspace("eval-1", "task-1", "KIS task")
    changed = store.set_avs_enabled(
        avs_enabled=True,
        expected_workspace_revision=initial.revision,
        user_id="member-1",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
    )

    reopened = WorkspaceStore(path)
    rebound = reopened.get_answer_workspace("eval-2", "task-2", "Other task")

    assert rebound.evaluation_id == "eval-2"
    assert rebound.task_scope_key == "task-2"
    assert rebound.avs_enabled is True
    assert rebound.revision > changed.workspace_revision
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    connection.close()


def test_frame_and_text_candidates_persist_with_optimistic_revisions(tmp_path: Path) -> None:
    """Preserve exact point milliseconds while revisions advance monotonically."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    state = store.get_answer_workspace("eval-1", "task-1", "KIS task")
    frame = store.add_frame_candidate(
        video_id="video.a",
        timestamp_ms=12_346,
        source_frame_id=None,
        user_id="member-1",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
        expected_workspace_revision=state.revision,
    )
    duplicate = store.add_frame_candidate(
        video_id="video.a",
        timestamp_ms=12_346,
        source_frame_id="canonical-frame",
        user_id="member-2",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
        expected_workspace_revision=frame.workspace_revision,
    )

    assert duplicate.candidate.candidate_id == frame.candidate.candidate_id
    assert duplicate.workspace_revision == frame.workspace_revision
    assert frame.candidate.timestamp_ms == 12_346
    assert frame.candidate.source_frame_id is None

    updated = store.update_frame_candidate(
        candidate_id=frame.candidate.candidate_id,
        video_id="video.a",
        timestamp_ms=12_500,
        expected_candidate_revision=1,
        expected_workspace_revision=frame.workspace_revision,
        user_id="member-2",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
    )
    text = store.add_text_candidate(
        text="a red car",
        user_id="member-1",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
        expected_workspace_revision=updated.workspace_revision,
    )

    assert updated.candidate.revision == 2
    assert updated.candidate.timestamp_ms == 12_500
    assert text.candidate.text == "a red car"
    assert text.candidate.revision == 1
    assert len(store.get_answer_workspace("eval-1", "task-1", "KIS task").candidates) == 2


def test_stale_workspace_revision_and_nonempty_task_switch_are_rejected(tmp_path: Path) -> None:
    """Require explicit candidate clearing before binding answers to a new task."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    initial = store.get_answer_workspace("eval-1", "task-1", "KIS task")
    added = store.add_frame_candidate(
        video_id="video-a",
        timestamp_ms=0,
        source_frame_id=None,
        user_id="member-1",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
        expected_workspace_revision=initial.revision,
    )

    with pytest.raises(AnswerWorkspaceConflict):
        store.add_text_candidate(
            text="stale",
            user_id="member-1",
            evaluation_id="eval-1",
            task_scope_key="task-1",
        task_name="KIS task",
            expected_workspace_revision=initial.revision,
        )
    mismatch = store.get_answer_workspace("eval-2", "task-2", "Other task")
    assert mismatch.task_scope_mismatch is True
    assert mismatch.evaluation_id == "eval-1"
    assert mismatch.active_evaluation_id == "eval-2"
    with pytest.raises(AnswerWorkspaceTaskScopeMismatch):
        store.add_text_candidate(
            text="wrong scope",
            user_id="member-1",
            evaluation_id="eval-2",
            task_scope_key="task-2",
            task_name="Other task",
            expected_workspace_revision=added.workspace_revision,
        )

    switched = store.clear_and_switch_task(
        expected_workspace_revision=added.workspace_revision,
        expected_old_evaluation_id="eval-1",
        expected_old_task_scope_key="task-1",
        target_evaluation_id="eval-2",
        target_task_scope_key="task-2",
        target_task_name="Other task",
        user_id="member-1",
    )

    assert switched.evaluation_id == "eval-2"
    assert switched.task_scope_key == "task-2"
    assert switched.candidates == []
