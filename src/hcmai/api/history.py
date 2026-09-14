"""SQLite persistence for replay history and the shared answer workspace.

This module owns versioned local state and optimistic transaction boundaries.
It does not perform DRES calls; DRES sessions and answer forwarding belong to
the VBS service and API router.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid

from hcmai.api.contracts.database import (
    DatabaseColumn,
    DatabaseQueryResponse,
    DatabaseRowsPage,
    DatabaseTable,
)
from hcmai.api.contracts.history import (
    FrameActivity,
    QueryHistoryCreate,
    QueryHistoryRecord,
)
from hcmai.api.contracts.workspace import (
    AnswerCandidate,
    AnswerCandidateMutation,
    AnswerModeMutation,
    AnswerWorkspaceSnapshot,
    SubmissionAttemptSummary,
)
from hcmai.common.utils.logging import get_logger


logger = get_logger(__name__)
_DATABASE_VERSION = 2
_DATABASE_TABLE_ORDER = {
    "query_history": "created_at DESC, rowid DESC",
    "answer_workspace_state": "singleton_id ASC",
    "answer_candidates": "created_at_ms ASC, candidate_id ASC",
}
_ANSWER_WORKSPACE_TABLES = frozenset({
    "answer_workspace_state",
    "answer_candidates",
    "submission_attempts",
})
_SQLITE_DML_ACTIONS = frozenset({
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
})
_SQLITE_ANSWER_INDEX_ACTIONS = frozenset({
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
})


class AnswerWorkspaceConflict(RuntimeError):
    """Reject a stale workspace/candidate revision or missing candidate."""


class AnswerWorkspaceTaskScopeMismatch(RuntimeError):
    """Reject using non-empty candidates against a different active DRES task."""

    def __init__(
        self,
        stored_evaluation_id: str | None,
        stored_task_scope_key: str | None,
        active_evaluation_id: str,
        active_task_scope_key: str,
    ) -> None:
        super().__init__(
            "TASK_SCOPE_MISMATCH: workspace is bound to a different DRES task"
        )
        self.stored_evaluation_id = stored_evaluation_id
        self.stored_task_scope_key = stored_task_scope_key
        self.active_evaluation_id = active_evaluation_id
        self.active_task_scope_key = active_task_scope_key


class AnswerSubmissionInFlight(RuntimeError):
    """Block workspace changes while a reserved DRES result is unresolved."""


@dataclass(frozen=True)
class SubmissionAttempt:
    """Immutable DRES request snapshot and its durable delivery state."""

    attempt_id: str
    kind: str
    evaluation_id: str
    task_scope_key: str
    task_name: str
    submitted_by_user_id: str
    workspace_revision: int
    snapshot_json: str
    state: str
    created_at_ms: int


class WorkspaceStore:
    """Persist query history and revision-checked answer candidates in SQLite."""

    def __init__(self, database_path: str | Path) -> None:
        """Create parent directories and migrate the configured database."""

        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_history(self, data: QueryHistoryCreate) -> QueryHistoryRecord:
        """Create one replay snapshot with empty viewed-frame activity."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO query_history (
                    query_id, user_id, query_text, result_snapshot_json,
                    viewed_frame_ids_json, created_at
                ) VALUES (?, ?, ?, ?, '[]', ?)
                """,
                (
                    data.query_id,
                    data.user_id,
                    data.query_text,
                    _json(data.result_snapshot),
                    _utc_now(),
                ),
            )
            row = _history_row(connection, data.query_id)
        logger.info("Query history created query_id=%s user_id=%s", data.query_id, data.user_id)
        return _history_record(row)

    def update_viewed_frame(self, query_id: str, frame_id: str) -> QueryHistoryRecord:
        """Append one viewed frame while preserving first-seen order."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _history_row(connection, query_id)
            frame_ids = _array(row["viewed_frame_ids_json"])
            _append_unique(frame_ids, [frame_id])
            connection.execute(
                "UPDATE query_history SET viewed_frame_ids_json = ? WHERE query_id = ?",
                (_json(frame_ids), query_id),
            )
            row = _history_row(connection, query_id)
        logger.info("Viewed frame recorded query_id=%s frame_id=%s", query_id, frame_id)
        return _history_record(row)

    def get_recent_history(self, user_id: str) -> list[QueryHistoryRecord]:
        """Return at most the newest twenty snapshots for one user."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM query_history
                WHERE user_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 20
                """,
                (user_id,),
            ).fetchall()
        records = [_history_record(row) for row in rows]
        logger.info("Query history loaded user_id=%s count=%d", user_id, len(records))
        return records

    def get_answer_workspace(
        self,
        active_evaluation_id: str,
        active_task_scope_key: str,
        active_task_name: str,
    ) -> AnswerWorkspaceSnapshot:
        """Hydrate candidates and show old/new scope when answers are ineligible."""

        _validate_scope(active_evaluation_id, active_task_scope_key)
        _validate_task_name(active_task_name)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = _workspace_state(connection)
            candidates = _candidate_rows(connection)
            pending = _pending_attempt(connection, state["pending_submission_id"])
            if not candidates and pending is None and (
                state["evaluation_id"] != active_evaluation_id
                or state["task_scope_key"] != active_task_scope_key
                or state["task_name"] != active_task_name
            ):
                connection.execute(
                    """
                    UPDATE answer_workspace_state
                    SET evaluation_id = ?, task_scope_key = ?, task_name = ?, revision = revision + 1,
                        updated_by_user_id = 'system', updated_at_ms = ?
                    WHERE singleton_id = 1
                    """,
                    (active_evaluation_id, active_task_scope_key, active_task_name, _now_ms()),
                )
                state = _workspace_state(connection)
            return _workspace_snapshot(
                connection,
                state,
                active_evaluation_id,
                active_task_scope_key,
                active_task_name,
            )

    def add_frame_candidate(
        self,
        *,
        video_id: str,
        timestamp_ms: int,
        source_frame_id: str | None,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
        expected_workspace_revision: int,
    ) -> AnswerModeMutation:
        """Add one exact FRAME answer, returning an existing duplicate unchanged."""

        _validate_candidate_identity(video_id, timestamp_ms)
        candidate_id = uuid.uuid4().hex
        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            existing = connection.execute(
                """
                SELECT * FROM answer_candidates
                WHERE kind = 'FRAME' AND video_id = ? AND timestamp_ms = ?
                """,
                (video_id, timestamp_ms),
            ).fetchone()
            if existing is not None:
                return AnswerCandidateMutation(
                    candidate=_candidate(existing),
                    workspace_revision=state["revision"],
                )
            connection.execute(
                """
                INSERT INTO answer_candidates (
                    candidate_id, kind, source_frame_id, video_id, timestamp_ms,
                    text, contributed_by_user_id, created_at_ms, revision,
                    submitted_at_ms, submitted_by_user_id, dres_status,
                    evaluation_id, task_scope_key
                ) VALUES (?, 'FRAME', ?, ?, ?, NULL, ?, ?, 1, NULL, NULL, NULL, ?, ?)
                """,
                (
                    candidate_id,
                    source_frame_id,
                    video_id,
                    timestamp_ms,
                    user_id,
                    now,
                    evaluation_id,
                    task_scope_key,
                ),
            )
            revision = self._bump_workspace(connection, user_id, now)
            row = _candidate_row(connection, candidate_id)
        return AnswerCandidateMutation(candidate=_candidate(row), workspace_revision=revision)

    def add_text_candidate(
        self,
        *,
        text: str,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
        expected_workspace_revision: int,
    ) -> AnswerCandidateMutation:
        """Add one plaintext VQA answer without deduplicating team proposals."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("TEXT candidate must not be blank")
        candidate_id = uuid.uuid4().hex
        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            connection.execute(
                """
                INSERT INTO answer_candidates (
                    candidate_id, kind, source_frame_id, video_id, timestamp_ms,
                    text, contributed_by_user_id, created_at_ms, revision,
                    submitted_at_ms, submitted_by_user_id, dres_status,
                    evaluation_id, task_scope_key
                ) VALUES (?, 'TEXT', NULL, NULL, NULL, ?, ?, ?, 1, NULL, NULL, NULL, ?, ?)
                """,
                (candidate_id, text, user_id, now, evaluation_id, task_scope_key),
            )
            revision = self._bump_workspace(connection, user_id, now)
            row = _candidate_row(connection, candidate_id)
        return AnswerCandidateMutation(candidate=_candidate(row), workspace_revision=revision)

    def update_frame_candidate(
        self,
        *,
        candidate_id: str,
        video_id: str,
        timestamp_ms: int,
        expected_candidate_revision: int,
        expected_workspace_revision: int,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
    ) -> AnswerCandidateMutation:
        """Edit a FRAME answer while preserving provenance and exact milliseconds."""

        _validate_candidate_identity(video_id, timestamp_ms)
        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            row = _matching_candidate(
                connection,
                candidate_id,
                "FRAME",
                expected_candidate_revision,
            )
            collision = connection.execute(
                """
                SELECT candidate_id FROM answer_candidates
                WHERE kind = 'FRAME' AND video_id = ? AND timestamp_ms = ?
                  AND candidate_id <> ?
                """,
                (video_id, timestamp_ms, candidate_id),
            ).fetchone()
            if collision is not None:
                raise AnswerWorkspaceConflict("A FRAME candidate already exists at this moment")
            connection.execute(
                """
                UPDATE answer_candidates
                SET video_id = ?, timestamp_ms = ?, revision = revision + 1
                WHERE candidate_id = ?
                """,
                (video_id, timestamp_ms, candidate_id),
            )
            revision = self._bump_workspace(connection, user_id, now)
            updated = _candidate_row(connection, row["candidate_id"])
        return AnswerCandidateMutation(candidate=_candidate(updated), workspace_revision=revision)

    def update_text_candidate(
        self,
        *,
        candidate_id: str,
        text: str,
        expected_candidate_revision: int,
        expected_workspace_revision: int,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
    ) -> AnswerCandidateMutation:
        """Edit one TEXT answer under candidate and workspace revision checks."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("TEXT candidate must not be blank")
        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            row = _matching_candidate(
                connection,
                candidate_id,
                "TEXT",
                expected_candidate_revision,
            )
            connection.execute(
                "UPDATE answer_candidates SET text = ?, revision = revision + 1 WHERE candidate_id = ?",
                (text, candidate_id),
            )
            revision = self._bump_workspace(connection, user_id, now)
            updated = _candidate_row(connection, row["candidate_id"])
        return AnswerCandidateMutation(candidate=_candidate(updated), workspace_revision=revision)

    def delete_candidate(
        self,
        *,
        candidate_id: str,
        expected_candidate_revision: int,
        expected_workspace_revision: int,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
    ) -> int:
        """Delete one candidate only when both optimistic revisions still match."""

        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            _matching_candidate(connection, candidate_id, None, expected_candidate_revision)
            connection.execute("DELETE FROM answer_candidates WHERE candidate_id = ?", (candidate_id,))
            return self._bump_workspace(connection, user_id, now)

    def clear_answer_candidates(
        self,
        *,
        expected_workspace_revision: int,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
    ) -> AnswerWorkspaceSnapshot:
        """Clear all current candidates with one revision-checked transaction."""

        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            connection.execute("DELETE FROM answer_candidates")
            self._bump_workspace(connection, user_id, now)
            state = _workspace_state(connection)
            return _workspace_snapshot(connection, state, evaluation_id, task_scope_key, task_name)

    def set_avs_enabled(
        self,
        *,
        avs_enabled: bool,
        expected_workspace_revision: int,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
    ) -> AnswerCandidateMutation:
        """Set shared AVS mode and increment revision only when it changes."""

        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            if bool(state["avs_enabled"]) != avs_enabled:
                connection.execute(
                    """
                    UPDATE answer_workspace_state SET avs_enabled = ?
                    WHERE singleton_id = 1
                    """,
                    (int(avs_enabled),),
                )
                self._bump_workspace(connection, user_id, now)
            state = _workspace_state(connection)
            return AnswerModeMutation(
                avs_enabled=bool(state["avs_enabled"]),
                workspace_revision=state["revision"],
            )

    def clear_and_switch_task(
        self,
        *,
        expected_workspace_revision: int,
        expected_old_evaluation_id: str,
        expected_old_task_scope_key: str,
        target_evaluation_id: str,
        target_task_scope_key: str,
        target_task_name: str,
        user_id: str,
    ) -> AnswerWorkspaceSnapshot:
        """Clear old-task answers and atomically bind the workspace to a new task."""

        _validate_scope(target_evaluation_id, target_task_scope_key)
        _validate_task_name(target_task_name)
        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = _workspace_state(connection)
            self._assert_not_reserved(state)
            self._assert_revision(state, expected_workspace_revision)
            if (
                state["evaluation_id"] != expected_old_evaluation_id
                or state["task_scope_key"] != expected_old_task_scope_key
            ):
                raise AnswerWorkspaceConflict("Workspace task scope changed before switching")
            connection.execute("DELETE FROM answer_candidates")
            connection.execute(
                """
                UPDATE answer_workspace_state
                SET evaluation_id = ?, task_scope_key = ?, task_name = ?, revision = revision + 1,
                    updated_by_user_id = ?, updated_at_ms = ?
                WHERE singleton_id = 1
                """,
                (target_evaluation_id, target_task_scope_key, target_task_name, user_id, now),
            )
            state = _workspace_state(connection)
            return _workspace_snapshot(
                connection,
                state,
                target_evaluation_id,
                target_task_scope_key,
                target_task_name,
            )

    def reserve_submission(
        self,
        *,
        kind: str,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
        expected_workspace_revision: int,
        candidates: list[dict[str, object]],
        dres_payload: dict[str, object],
    ) -> SubmissionAttempt:
        """Atomically freeze eligible answer revisions and exact DRES JSON."""

        _validate_scope(evaluation_id, task_scope_key)
        _validate_task_name(task_name)
        _validate_submission_payload(kind, task_name, candidates, dres_payload)
        now = _now_ms()
        attempt_id = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            state = self._prepare_mutation(
                connection,
                user_id,
                evaluation_id,
                task_scope_key,
                task_name,
                expected_workspace_revision,
            )
            if kind in {"KIS", "VQA"} and bool(state["avs_enabled"]):
                raise AnswerWorkspaceConflict("KIS/VQA submission requires AVS mode to be off")
            if kind == "AVS" and not bool(state["avs_enabled"]):
                raise AnswerWorkspaceConflict("AVS submission requires AVS mode to be on")

            candidate_rows = connection.execute(
                "SELECT * FROM answer_candidates WHERE submitted_at_ms IS NULL "
                "ORDER BY created_at_ms ASC, candidate_id ASC"
            ).fetchall()
            expected_kind = "TEXT" if kind == "VQA" else "FRAME"
            eligible = [row for row in candidate_rows if row["kind"] == expected_kind]
            selected = _validate_candidate_selection(kind, candidates, eligible)
            _validate_payload_matches_candidates(kind, selected, dres_payload)
            candidate_revisions = [
                {"candidate_id": row["candidate_id"], "revision": row["revision"]}
                for row in selected
            ]
            snapshot = {
                "kind": kind,
                "evaluation_id": evaluation_id,
                "task_scope_key": task_scope_key,
                "task_name": task_name,
                "submitted_by_user_id": user_id,
                "workspace_revision": int(state["revision"]),
                "candidate_revisions": candidate_revisions,
                "dres_payload": dres_payload,
            }
            snapshot_json = json.dumps(
                snapshot,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            connection.execute(
                """
                INSERT INTO submission_attempts (
                  attempt_id, kind, evaluation_id, task_scope_key, task_name,
                  submitted_by_user_id, workspace_revision, snapshot_json, state, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'FORWARDING', ?)
                """,
                (
                    attempt_id,
                    kind,
                    evaluation_id,
                    task_scope_key,
                    task_name,
                    user_id,
                    state["revision"],
                    snapshot_json,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE answer_workspace_state
                SET pending_submission_id = ?, pending_submission_state = 'FORWARDING',
                    revision = revision + 1, updated_by_user_id = ?, updated_at_ms = ?
                WHERE singleton_id = 1
                """,
                (attempt_id, user_id, now),
            )
            row = connection.execute(
                "SELECT * FROM submission_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        return _submission_attempt(row)

    def get_submission_attempt(self, attempt_id: str) -> SubmissionAttempt:
        """Read a durable attempt by ID without exposing credentials or sessions."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM submission_attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Submission attempt {attempt_id!r} not found")
        return _submission_attempt(row)

    def mark_submission_unknown(self, attempt_id: str) -> AnswerWorkspaceSnapshot:
        """Keep the in-flight lock when DRES acceptance cannot be determined."""

        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = _attempt_row(connection, attempt_id)
            state = _workspace_state(connection)
            if attempt["state"] != "FORWARDING" or state["pending_submission_id"] != attempt_id:
                raise AnswerWorkspaceConflict("Only the current FORWARDING attempt can become UNKNOWN")
            connection.execute(
                "UPDATE submission_attempts SET state = 'UNKNOWN' WHERE attempt_id = ?",
                (attempt_id,),
            )
            connection.execute(
                """
                UPDATE answer_workspace_state
                SET pending_submission_state = 'UNKNOWN', revision = revision + 1,
                    updated_by_user_id = 'system', updated_at_ms = ?
                WHERE singleton_id = 1
                """,
                (now,),
            )
            return _workspace_snapshot(
                connection,
                _workspace_state(connection),
                attempt["evaluation_id"],
                attempt["task_scope_key"],
                attempt["task_name"],
            )

    def complete_submission(
        self,
        attempt_id: str,
        *,
        accepted: bool,
        dres_status: str,
    ) -> AnswerWorkspaceSnapshot:
        """Commit a definitive DRES acceptance or rejection for one send."""

        return self._finalize_submission(
            attempt_id,
            accepted=accepted,
            dres_status=dres_status,
            required_state="FORWARDING",
        )

    def resolve_unknown_submission(
        self,
        attempt_id: str,
        *,
        accepted: bool,
    ) -> AnswerWorkspaceSnapshot:
        """Resolve an ambiguous attempt only from its frozen durable snapshot."""

        return self._finalize_submission(
            attempt_id,
            accepted=accepted,
            dres_status="accepted" if accepted else "not accepted",
            required_state="UNKNOWN",
        )

    def _finalize_submission(
        self,
        attempt_id: str,
        *,
        accepted: bool,
        dres_status: str,
        required_state: str,
    ) -> AnswerWorkspaceSnapshot:
        """Release the matching lock and update exactly frozen candidate revisions."""

        now = _now_ms()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = _attempt_row(connection, attempt_id)
            state = _workspace_state(connection)
            if attempt["state"] != required_state or state["pending_submission_id"] != attempt_id:
                raise AnswerWorkspaceConflict(
                    f"Only the current {required_state} attempt can be resolved"
                )
            frozen = json.loads(attempt["snapshot_json"])
            if accepted:
                for candidate in frozen["candidate_revisions"]:
                    row = _matching_candidate(
                        connection,
                        candidate["candidate_id"],
                        None,
                        candidate["revision"],
                    )
                    if row["evaluation_id"] != attempt["evaluation_id"] or row["task_scope_key"] != attempt["task_scope_key"]:
                        raise AnswerWorkspaceConflict("Frozen answer task scope changed")
                    connection.execute(
                        """
                        UPDATE answer_candidates
                        SET submitted_at_ms = ?, submitted_by_user_id = ?, dres_status = ?
                        WHERE candidate_id = ? AND revision = ? AND submitted_at_ms IS NULL
                        """,
                        (
                            now,
                            attempt["submitted_by_user_id"],
                            dres_status[:120],
                            candidate["candidate_id"],
                            candidate["revision"],
                        ),
                    )
            connection.execute(
                """
                UPDATE submission_attempts
                SET state = ?, resolved_at_ms = ?
                WHERE attempt_id = ?
                """,
                ("ACCEPTED" if accepted else "NOT_ACCEPTED", now, attempt_id),
            )
            connection.execute(
                """
                UPDATE answer_workspace_state
                SET pending_submission_id = NULL, pending_submission_state = NULL,
                    revision = revision + 1, updated_by_user_id = ?, updated_at_ms = ?
                WHERE singleton_id = 1
                """,
                (attempt["submitted_by_user_id"], now),
            )
            return _workspace_snapshot(
                connection,
                _workspace_state(connection),
                attempt["evaluation_id"],
                attempt["task_scope_key"],
                attempt["task_name"],
            )

    def list_database_tables(self) -> list[DatabaseTable]:
        """Describe the explicitly allowlisted SQLite tables."""

        with self._readonly_connection() as connection:
            return [
                DatabaseTable(
                    name=table_name,
                    row_count=connection.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0],
                    columns=_table_columns(connection, table_name),
                )
                for table_name in _DATABASE_TABLE_ORDER
            ]

    def list_database_rows(
        self,
        table_name: str,
        *,
        page: int,
        page_size: int,
    ) -> DatabaseRowsPage:
        """Read one stable page from an allowlisted application table."""

        try:
            order_by = _DATABASE_TABLE_ORDER[table_name]
        except KeyError:
            raise KeyError(f"Database table {table_name!r} is not available") from None
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("Database pagination is outside the supported bounds")
        with self._readonly_connection() as connection:
            total_rows = int(
                connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            )
            offset = (page - 1) * page_size
            rows = connection.execute(
                f"SELECT * FROM {table_name} ORDER BY {order_by} LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
        return DatabaseRowsPage(
            table=table_name,
            page=page,
            page_size=page_size,
            total_rows=total_rows,
            total_pages=(total_rows + page_size - 1) // page_size,
            rows=[dict(row) for row in rows],
        )

    def execute_query(self, query: str, *, max_rows: int = 100) -> DatabaseQueryResponse:
        """Execute database-console SQL without bypassing typed answer mutations."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("SQL query cannot be empty")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        start_time = time.perf_counter()
        denied_tables: list[str] = []

        def authorize_answer_table_writes(
            action: int,
            table_name: str | None,
            second_argument: str | None,
            _database_name: str | None,
            _trigger_name: str | None,
        ) -> int:
            """Keep direct SQL from mutating revision-owned answer state."""

            protected_table = (
                second_argument
                if action == sqlite3.SQLITE_ALTER_TABLE
                else table_name
            )
            if (
                action == sqlite3.SQLITE_ALTER_TABLE or action in _SQLITE_DML_ACTIONS
            ) and protected_table is not None and protected_table.casefold() in _ANSWER_WORKSPACE_TABLES:
                denied_tables.append(protected_table)
                return sqlite3.SQLITE_DENY
            if (
                action in _SQLITE_ANSWER_INDEX_ACTIONS
                and second_argument is not None
                and second_argument.casefold() in _ANSWER_WORKSPACE_TABLES
            ):
                denied_tables.append(second_argument)
                return sqlite3.SQLITE_DENY
            if action in {
                sqlite3.SQLITE_DROP_INDEX,
                sqlite3.SQLITE_DROP_TEMP_INDEX,
                sqlite3.SQLITE_DROP_TRIGGER,
                sqlite3.SQLITE_DROP_TEMP_TRIGGER,
            } and table_name is not None and table_name.casefold() in protected_auxiliary_objects:
                denied_tables.append(table_name)
                return sqlite3.SQLITE_DENY
            if (
                action == sqlite3.SQLITE_DROP_VTABLE
                and table_name is not None
                and table_name.casefold() in _ANSWER_WORKSPACE_TABLES
            ):
                denied_tables.append(table_name)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        try:
            with self._connection() as connection:
                protected_auxiliary_objects = {
                    str(row[0]).casefold()
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type IN ('index', 'trigger') "
                        "AND lower(tbl_name) IN "
                        "('answer_workspace_state', 'answer_candidates', 'submission_attempts')"
                    ).fetchall()
                }
                connection.set_authorizer(authorize_answer_table_writes)
                cursor = connection.execute(cleaned_query)
                if cursor.description is not None:
                    columns = [column[0] for column in cursor.description]
                    rows = [dict(row) for row in cursor.fetchmany(max_rows)]
                    is_mutation = False
                    rows_affected = 0
                else:
                    columns = []
                    rows = []
                    is_mutation = True
                    rows_affected = cursor.rowcount if cursor.rowcount >= 0 else 0
        except sqlite3.Error as error:
            if denied_tables:
                raise ValueError(
                    "Answer workspace tables are managed by revision-checked workspace commands"
                ) from error
            raise ValueError(f"SQLite execution failed: {error}") from error
        return DatabaseQueryResponse(
            query=cleaned_query,
            columns=columns,
            rows=rows,
            rows_affected=rows_affected,
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 3),
            is_mutation=is_mutation,
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open one short SQLite transaction and roll back every failed operation."""

        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _readonly_connection(self) -> Iterator[sqlite3.Connection]:
        """Open SQLite in URI read-only mode for database-browser requests."""

        database_uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        try:
            yield connection
        finally:
            connection.close()

    def _prepare_mutation(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        evaluation_id: str,
        task_scope_key: str,
        task_name: str,
        expected_workspace_revision: int,
    ) -> sqlite3.Row:
        """Check reservation, scope, and revision before any candidate write."""

        _validate_scope(evaluation_id, task_scope_key)
        _validate_task_name(task_name)
        state = _workspace_state(connection)
        self._assert_not_reserved(state)
        candidate_count = int(
            connection.execute("SELECT COUNT(*) FROM answer_candidates").fetchone()[0]
        )
        if candidate_count and (
            state["evaluation_id"] != evaluation_id or state["task_scope_key"] != task_scope_key
        ):
            raise AnswerWorkspaceTaskScopeMismatch(
                state["evaluation_id"],
                state["task_scope_key"],
                evaluation_id,
                task_scope_key,
            )
        if not candidate_count and (
            state["evaluation_id"] != evaluation_id or state["task_scope_key"] != task_scope_key
        ):
            connection.execute(
                """
                UPDATE answer_workspace_state
                SET evaluation_id = ?, task_scope_key = ?, task_name = ?, revision = revision + 1,
                    updated_by_user_id = ?, updated_at_ms = ?
                WHERE singleton_id = 1
                """,
                (evaluation_id, task_scope_key, task_name, user_id, _now_ms()),
            )
            state = _workspace_state(connection)
        self._assert_revision(state, expected_workspace_revision)
        return state

    @staticmethod
    def _assert_revision(state: sqlite3.Row, expected: int) -> None:
        """Require the client revision to match the serialized workspace state."""

        if state["revision"] != expected:
            raise AnswerWorkspaceConflict(
                f"Workspace revision changed from {expected} to {state['revision']}"
            )

    @staticmethod
    def _assert_not_reserved(state: sqlite3.Row) -> None:
        """Reject edits while a DRES send has an unresolved outcome."""

        if state["pending_submission_id"] is not None:
            raise AnswerSubmissionInFlight("SUBMISSION_IN_FLIGHT")

    @staticmethod
    def _bump_workspace(
        connection: sqlite3.Connection,
        user_id: str,
        updated_at_ms: int,
    ) -> int:
        """Increment the shared revision after a committed semantic mutation."""

        connection.execute(
            """
            UPDATE answer_workspace_state
            SET revision = revision + 1, updated_by_user_id = ?, updated_at_ms = ?
            WHERE singleton_id = 1
            """,
            (user_id, updated_at_ms),
        )
        return int(_workspace_state(connection)["revision"])

    @staticmethod
    def _create_answer_tables(connection: sqlite3.Connection) -> None:
        """Create all durable answer, reservation, and attempt tables."""

        statements = [
            """
            CREATE TABLE IF NOT EXISTS answer_workspace_state (
              singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
              avs_enabled INTEGER NOT NULL CHECK (avs_enabled IN (0, 1)),
              evaluation_id TEXT,
              task_scope_key TEXT,
              task_name TEXT,
              revision INTEGER NOT NULL CHECK (revision >= 0),
              pending_submission_id TEXT,
              pending_submission_state TEXT CHECK (
                pending_submission_state IS NULL
                OR pending_submission_state IN ('FORWARDING', 'UNKNOWN')
              ),
              updated_by_user_id TEXT NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              CHECK ((evaluation_id IS NULL) = (task_scope_key IS NULL)),
              CHECK ((task_scope_key IS NULL) = (task_name IS NULL)),
              CHECK ((pending_submission_id IS NULL) = (pending_submission_state IS NULL))
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS answer_candidates (
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
              task_scope_key TEXT NOT NULL,
              CHECK (
                (kind = 'FRAME' AND video_id IS NOT NULL AND timestamp_ms IS NOT NULL AND text IS NULL)
                OR
                (kind = 'TEXT' AND text IS NOT NULL AND source_frame_id IS NULL
                  AND video_id IS NULL AND timestamp_ms IS NULL)
              )
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS answer_frame_unique
            ON answer_candidates(video_id, timestamp_ms) WHERE kind = 'FRAME'
            """,
            """
            CREATE TABLE IF NOT EXISTS submission_attempts (
              attempt_id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('KIS', 'VQA', 'AVS')),
              evaluation_id TEXT NOT NULL,
              task_scope_key TEXT NOT NULL,
              task_name TEXT NOT NULL,
              submitted_by_user_id TEXT NOT NULL,
              workspace_revision INTEGER NOT NULL,
              snapshot_json TEXT NOT NULL,
              state TEXT NOT NULL CHECK (
                state IN ('FORWARDING', 'UNKNOWN', 'ACCEPTED', 'NOT_ACCEPTED')
              ),
              created_at_ms INTEGER NOT NULL,
              resolved_at_ms INTEGER
            )
            """,
        ]
        for statement in statements:
            connection.execute(statement)
        connection.execute(
            """
            INSERT INTO answer_workspace_state (
              singleton_id, avs_enabled, evaluation_id, task_scope_key, task_name, revision,
              pending_submission_id, pending_submission_state,
              updated_by_user_id, updated_at_ms
            ) VALUES (1, 0, NULL, NULL, NULL, 0, NULL, NULL, 'system', ?)
            ON CONFLICT(singleton_id) DO NOTHING
            """,
            (_now_ms(),),
        )

    def _initialize(self) -> None:
        """Apply one atomic versioned migration from the legacy file schema."""

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _DATABASE_VERSION:
                raise RuntimeError(
                    f"Workspace database version {version} is newer than supported {_DATABASE_VERSION}"
                )
            if version == 0:
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "query_history" in tables:
                    columns = {
                        str(row[1])
                        for row in connection.execute("PRAGMA table_info(query_history)")
                    }
                    if "submission_file_names_json" in columns or "submitted_frame_ids_json" in columns:
                        connection.execute(
                            "ALTER TABLE query_history RENAME TO query_history_legacy"
                        )
                        self._create_query_history_table(connection)
                        connection.execute(
                            """
                            INSERT INTO query_history (
                              query_id, user_id, query_text, result_snapshot_json,
                              viewed_frame_ids_json, created_at
                            )
                            SELECT query_id, user_id, query_text, result_snapshot_json,
                              viewed_frame_ids_json, created_at
                            FROM query_history_legacy
                            """
                        )
                        connection.execute("DROP TABLE query_history_legacy")
                else:
                    self._create_query_history_table(connection)
                connection.execute("DROP TABLE IF EXISTS submission_files")
                self._create_answer_tables(connection)
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS query_history_user_created_idx
                    ON query_history (user_id, created_at DESC)
                    """
                )
                connection.execute(f"PRAGMA user_version = {_DATABASE_VERSION}")
            elif version == 1:
                self._create_query_history_table(connection)
                self._migrate_answer_workspace_v1(connection)
                connection.execute(f"PRAGMA user_version = {_DATABASE_VERSION}")
            else:
                self._create_query_history_table(connection)
                self._create_answer_tables(connection)
            self._recover_forwarding_attempts(connection)
            logger.info("Workspace database ready path=%s version=%d", self.database_path, _DATABASE_VERSION)

    @classmethod
    def _migrate_answer_workspace_v1(cls, connection: sqlite3.Connection) -> None:
        """Rebuild v1 task-ID tables while preserving values and frozen audit JSON."""

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not _ANSWER_WORKSPACE_TABLES <= tables:
            raise RuntimeError("Version 1 database is missing answer-workspace tables")

        # SQLite keeps an index name attached to its original table after ALTER TABLE.
        # Drop it first so _create_answer_tables can install it on the replacement.
        connection.execute("DROP INDEX IF EXISTS answer_frame_unique")
        for table_name in sorted(_ANSWER_WORKSPACE_TABLES):
            connection.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_v1")

        cls._create_answer_tables(connection)
        connection.execute("DELETE FROM answer_workspace_state")
        connection.execute(
            """
            INSERT INTO answer_workspace_state (
              singleton_id, avs_enabled, evaluation_id, task_scope_key, task_name,
              revision, pending_submission_id, pending_submission_state,
              updated_by_user_id, updated_at_ms
            )
            SELECT singleton_id, avs_enabled, evaluation_id, task_id,
              CASE WHEN task_id IS NULL THEN NULL ELSE 'legacy-unverified' END,
              revision, pending_submission_id, pending_submission_state,
              updated_by_user_id, updated_at_ms
            FROM answer_workspace_state_v1
            """
        )
        connection.execute(
            """
            INSERT INTO answer_candidates (
              candidate_id, kind, source_frame_id, video_id, timestamp_ms, text,
              contributed_by_user_id, created_at_ms, revision, submitted_at_ms,
              submitted_by_user_id, dres_status, evaluation_id, task_scope_key
            )
            SELECT candidate_id, kind, source_frame_id, video_id, timestamp_ms, text,
              contributed_by_user_id, created_at_ms, revision, submitted_at_ms,
              submitted_by_user_id, dres_status, evaluation_id, task_id
            FROM answer_candidates_v1
            """
        )
        connection.execute(
            """
            INSERT INTO submission_attempts (
              attempt_id, kind, evaluation_id, task_scope_key, task_name,
              submitted_by_user_id, workspace_revision, snapshot_json, state,
              created_at_ms, resolved_at_ms
            )
            SELECT attempt_id, kind, evaluation_id, task_id, 'legacy-unverified',
              submitted_by_user_id, workspace_revision, snapshot_json, state,
              created_at_ms, resolved_at_ms
            FROM submission_attempts_v1
            """
        )
        for table_name in sorted(_ANSWER_WORKSPACE_TABLES):
            connection.execute(f"DROP TABLE {table_name}_v1")

    @staticmethod
    def _recover_forwarding_attempts(connection: sqlite3.Connection) -> None:
        """Convert interrupted sends to UNKNOWN without releasing their lock."""

        connection.execute(
            "UPDATE submission_attempts SET state = 'UNKNOWN' WHERE state = 'FORWARDING'"
        )
        connection.execute(
            """
            UPDATE answer_workspace_state
            SET pending_submission_state = 'UNKNOWN'
            WHERE pending_submission_id IN (
              SELECT attempt_id FROM submission_attempts WHERE state = 'UNKNOWN'
            ) AND pending_submission_state = 'FORWARDING'
            """
        )

    @staticmethod
    def _create_query_history_table(connection: sqlite3.Connection) -> None:
        """Create the current replay history schema without submission fields."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
              query_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              query_text TEXT NOT NULL,
              result_snapshot_json TEXT NOT NULL,
              viewed_frame_ids_json TEXT NOT NULL DEFAULT '[]',
              created_at TEXT NOT NULL
            )
            """
        )


def _workspace_state(connection: sqlite3.Connection) -> sqlite3.Row:
    """Read the singleton workspace row or raise if schema is damaged."""

    row = connection.execute(
        "SELECT * FROM answer_workspace_state WHERE singleton_id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("Answer workspace state row is missing")
    return row


def _candidate_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return all candidates in the stable order used by clients and AVS."""

    return connection.execute(
        "SELECT * FROM answer_candidates ORDER BY created_at_ms ASC, candidate_id ASC"
    ).fetchall()


def _pending_attempt(
    connection: sqlite3.Connection,
    attempt_id: str | None,
) -> SubmissionAttemptSummary | None:
    """Load only the safe status fields for one unresolved reservation."""

    if attempt_id is None:
        return None
    row = connection.execute(
        "SELECT attempt_id, kind, state, submitted_by_user_id, created_at_ms "
        "FROM submission_attempts WHERE attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    return None if row is None else SubmissionAttemptSummary.model_validate(dict(row))


def _attempt_row(connection: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
    """Load a durable submission attempt or reject the unknown identifier."""

    row = connection.execute(
        "SELECT * FROM submission_attempts WHERE attempt_id = ?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Submission attempt {attempt_id!r} not found")
    return row


def _submission_attempt(row: sqlite3.Row) -> SubmissionAttempt:
    """Convert one durable attempt while preserving its immutable JSON bytes."""

    return SubmissionAttempt(
        attempt_id=row["attempt_id"],
        kind=row["kind"],
        evaluation_id=row["evaluation_id"],
        task_scope_key=row["task_scope_key"],
        task_name=row["task_name"],
        submitted_by_user_id=row["submitted_by_user_id"],
        workspace_revision=row["workspace_revision"],
        snapshot_json=row["snapshot_json"],
        state=row["state"],
        created_at_ms=row["created_at_ms"],
    )


def _validate_submission_payload(
    kind: str,
    task_name: str,
    candidates: list[dict[str, object]],
    payload: dict[str, object],
) -> None:
    """Fail closed on incomplete task names, empty answers, or wrong wire shape."""

    if kind not in {"KIS", "VQA", "AVS"}:
        raise AnswerWorkspaceConflict("Unsupported DRES submission mode")
    if not isinstance(task_name, str) or not task_name.strip():
        raise AnswerWorkspaceConflict("DRES task name is required for submission")
    if not isinstance(candidates, list) or not candidates:
        raise AnswerWorkspaceConflict("DRES submission requires at least one candidate")
    if not isinstance(payload, dict) or set(payload) != {"answerSets"}:
        raise AnswerWorkspaceConflict("DRES submission payload must contain answerSets only")
    answer_sets = payload["answerSets"]
    if not isinstance(answer_sets, list) or len(answer_sets) != 1:
        raise AnswerWorkspaceConflict("Submission must contain one answerSet")
    answer_set = answer_sets[0]
    if (
        not isinstance(answer_set, dict)
        or set(answer_set) != {"taskName", "answers"}
        or answer_set.get("taskName") != task_name
        or not isinstance(answer_set.get("answers"), list)
        or not answer_set["answers"]
    ):
        raise AnswerWorkspaceConflict("DRES answerSet requires the live task name and non-empty answers")
    answers = answer_set["answers"]
    if len(answers) != len(candidates):
        raise AnswerWorkspaceConflict("DRES answer count does not match the frozen candidates")
    if kind in {"KIS", "VQA"} and len(answers) != 1:
        raise AnswerWorkspaceConflict("KIS and VQA submit exactly one answer")
    for answer in answers:
        if kind == "VQA":
            if (
                not isinstance(answer, dict)
                or set(answer) != {"text"}
                or not isinstance(answer.get("text"), str)
                or not answer["text"].strip()
            ):
                raise AnswerWorkspaceConflict("VQA answers must contain non-blank text only")
        elif (
            not isinstance(answer, dict)
            or set(answer) != {"mediaItemName", "start", "end"}
            or not isinstance(answer.get("mediaItemName"), str)
            or not answer["mediaItemName"].strip()
            or type(answer.get("start")) is not int
            or type(answer.get("end")) is not int
            or answer["start"] < 0
            or answer["start"] != answer["end"]
        ):
            raise AnswerWorkspaceConflict("KIS/AVS answers must use an exact temporal point")


def _validate_candidate_selection(
    kind: str,
    candidates: list[dict[str, object]],
    eligible: list[sqlite3.Row],
) -> list[sqlite3.Row]:
    """Match the submitted membership and order against live eligible rows."""

    if kind not in {"KIS", "VQA", "AVS"}:
        raise AnswerWorkspaceConflict("Unsupported DRES submission mode")
    if not isinstance(candidates, list) or not candidates:
        raise AnswerWorkspaceConflict("Submission candidate set must not be empty")
    ids: list[str] = []
    for item in candidates:
        if (
            not isinstance(item, dict)
            or set(item) != {"candidate_id", "expected_revision"}
            or not isinstance(item.get("candidate_id"), str)
            or not item["candidate_id"].strip()
            or type(item.get("expected_revision")) is not int
            or item["expected_revision"] < 1
        ):
            raise AnswerWorkspaceConflict("Candidate ID and revision are required")
        ids.append(item["candidate_id"])
    if len(ids) != len(set(ids)):
        raise AnswerWorkspaceConflict("Submission candidate IDs must be unique")

    eligible_by_id = {row["candidate_id"]: row for row in eligible}
    selected: list[sqlite3.Row] = []
    for item in candidates:
        row = eligible_by_id.get(item["candidate_id"])
        if row is None:
            raise AnswerWorkspaceConflict("Submission candidate is missing, ineligible, or already submitted")
        if row["revision"] != item["expected_revision"]:
            raise AnswerWorkspaceConflict("Submission candidate revision is stale")
        selected.append(row)

    if kind == "AVS":
        expected_ids = [row["candidate_id"] for row in eligible]
        if ids != expected_ids:
            raise AnswerWorkspaceConflict("AVS membership and order must match every eligible FRAME candidate")
    elif len(selected) != 1:
        raise AnswerWorkspaceConflict("KIS/VQA submission requires exactly one candidate")
    return selected


def _validate_payload_matches_candidates(
    kind: str,
    candidates: list[sqlite3.Row],
    payload: dict[str, object],
) -> None:
    """Keep each payload answer tied to the selected immutable candidate value."""

    answer_set = payload["answerSets"][0]  # type: ignore[index]
    answers = answer_set["answers"]
    for row, answer in zip(candidates, answers, strict=True):
        if kind == "VQA":
            if answer["text"] != row["text"]:
                raise AnswerWorkspaceConflict("VQA payload text differs from the selected candidate")
        elif answer["start"] != row["timestamp_ms"] or answer["end"] != row["timestamp_ms"]:
            raise AnswerWorkspaceConflict("Temporal payload point differs from the selected candidate")


def _workspace_snapshot(
    connection: sqlite3.Connection,
    state: sqlite3.Row,
    active_evaluation_id: str,
    active_task_scope_key: str,
    active_task_name: str,
) -> AnswerWorkspaceSnapshot:
    """Materialize state, ordered candidate rows, and current task eligibility."""

    return AnswerWorkspaceSnapshot(
        avs_enabled=bool(state["avs_enabled"]),
        evaluation_id=state["evaluation_id"],
        task_scope_key=state["task_scope_key"],
        task_name=state["task_name"],
        revision=state["revision"],
        updated_by_user_id=state["updated_by_user_id"],
        updated_at_ms=state["updated_at_ms"],
        candidates=[_candidate(row) for row in _candidate_rows(connection)],
        pending_submission=_pending_attempt(connection, state["pending_submission_id"]),
        active_evaluation_id=active_evaluation_id,
        active_task_scope_key=active_task_scope_key,
        active_task_name=active_task_name,
        task_scope_mismatch=(
            state["evaluation_id"] not in (None, active_evaluation_id)
            or state["task_scope_key"] not in (None, active_task_scope_key)
        ),
    )


def _candidate(row: sqlite3.Row) -> AnswerCandidate:
    """Validate one stored row against the typed FRAME/TEXT invariant."""

    return AnswerCandidate.model_validate(dict(row))


def _candidate_row(connection: sqlite3.Connection, candidate_id: str) -> sqlite3.Row:
    """Fetch one candidate or reject a stale/missing client reference."""

    row = connection.execute(
        "SELECT * FROM answer_candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise AnswerWorkspaceConflict("Answer candidate no longer exists")
    return row


def _matching_candidate(
    connection: sqlite3.Connection,
    candidate_id: str,
    kind: str | None,
    expected_revision: int,
) -> sqlite3.Row:
    """Check candidate existence, optional kind, and optimistic revision."""

    row = _candidate_row(connection, candidate_id)
    if kind is not None and row["kind"] != kind:
        raise AnswerWorkspaceConflict(f"Candidate is not a {kind} answer")
    if row["revision"] != expected_revision:
        raise AnswerWorkspaceConflict(
            f"Candidate revision changed from {expected_revision} to {row['revision']}"
        )
    if row["submitted_at_ms"] is not None:
        raise AnswerWorkspaceConflict("Submitted candidates are immutable")
    return row


def _validate_scope(evaluation_id: str, task_scope_key: str) -> None:
    """Reject empty active evaluation and internal scope values."""

    if not isinstance(evaluation_id, str) or not evaluation_id.strip():
        raise ValueError("evaluation_id must not be blank")
    if not isinstance(task_scope_key, str) or not task_scope_key.strip():
        raise ValueError("task_scope_key must not be blank")


def _validate_task_name(task_name: str) -> None:
    """Require the exact non-blank official task name stored with a scope."""

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must not be blank")


def _validate_candidate_identity(video_id: str, timestamp_ms: int) -> None:
    """Validate the source ID and exact non-negative millisecond value."""

    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must not be blank")
    if type(timestamp_ms) is not int or timestamp_ms < 0:
        raise ValueError("timestamp_ms must be a non-negative integer")


def _history_row(connection: sqlite3.Connection, query_id: str) -> sqlite3.Row:
    """Load one query replay record or report its missing ID."""

    row = connection.execute(
        "SELECT * FROM query_history WHERE query_id = ?",
        (query_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Query history {query_id!r} not found")
    return row


def _history_record(row: sqlite3.Row) -> QueryHistoryRecord:
    """Convert a SQLite replay row without submission-file coupling."""

    return QueryHistoryRecord(
        query_id=row["query_id"],
        query_text=row["query_text"],
        result_snapshot=json.loads(row["result_snapshot_json"]),
        frame_activity=FrameActivity(
            viewed_frame_ids=_array(row["viewed_frame_ids_json"]),
        ),
    )


def _utc_now() -> str:
    """Return a sortable UTC timestamp for query replay history."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _now_ms() -> int:
    """Return wall-clock epoch milliseconds for workspace activity."""

    return int(time.time() * 1000)


def _json(value: object) -> str:
    """Encode compact UTF-8 JSON for SQLite snapshots."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _array(value: str) -> list[str]:
    """Decode one stored JSON string array."""

    return list(json.loads(value))


def _append_unique(target: list[str], values: list[str]) -> bool:
    """Append unseen frame IDs while preserving first-seen order."""

    changed = False
    for value in values:
        if value not in target:
            target.append(value)
            changed = True
    return changed


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[DatabaseColumn]:
    """Project schema facts for one already-allowlisted table name."""

    return [
        DatabaseColumn(
            name=row["name"],
            type=row["type"],
            nullable=not bool(row["notnull"]) and not bool(row["pk"]),
            primary_key=bool(row["pk"]),
        )
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    ]


__all__ = [
    "AnswerSubmissionInFlight",
    "AnswerWorkspaceConflict",
    "AnswerWorkspaceTaskScopeMismatch",
    "SubmissionAttempt",
    "WorkspaceStore",
]
