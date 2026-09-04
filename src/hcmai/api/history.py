"""SQLite persistence for query history and shared submission files.

This module stores replay snapshots and collaborative file state. Canonical
frame validation remains at the API boundary where SearchService is available.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

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
    SubmissionFile,
)
from hcmai.common.utils.logging import get_logger


logger = get_logger(__name__)

_DATABASE_TABLE_ORDER = {
    "query_history": "created_at DESC, rowid DESC",
    "submission_files": "name ASC",
}


class RevisionConflict(Exception):
    """Report the latest file when an optimistic-lock revision is stale."""

    def __init__(self, file: SubmissionFile) -> None:
        super().__init__(f"Submission file revision changed: {file.name}")
        self.file = file


class WorkspaceStore:
    """Persist query history and shared submission files in one SQLite DB."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()


    def create_history(self, data: QueryHistoryCreate) -> QueryHistoryRecord:
        """Create one replay snapshot with empty activity arrays."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO query_history (
                    query_id, user_id, query_text, result_snapshot_json,
                    submission_file_names_json, viewed_frame_ids_json,
                    submitted_frame_ids_json, created_at
                ) VALUES (?, ?, ?, ?, '[]', '[]', '[]', ?)
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

        logger.info(
            "Query history created query_id=%s user_id=%s",
            data.query_id,
            data.user_id,
        )
        return _history_record(row)


    def update_viewed_frame(
        self,
        query_id: str,
        frame_id: str,
    ) -> QueryHistoryRecord:
        """Append one viewed frame while preserving first-seen order."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _history_row(connection, query_id)
            frame_ids = _array(row["viewed_frame_ids_json"])
            changed = _append_unique(frame_ids, [frame_id])
            connection.execute(
                "UPDATE query_history SET viewed_frame_ids_json = ? "
                "WHERE query_id = ?",
                (_json(frame_ids), query_id),
            )
            row = _history_row(connection, query_id)

        logger.info(
            "Viewed frame recorded query_id=%s frame_id=%s changed=%s",
            query_id,
            frame_id,
            changed,
        )
        return _history_record(row)


    def update_submission(
        self,
        query_id: str,
        file_name: str,
        submission_line: str,
        frame_ids: list[str],
    ) -> QueryHistoryRecord:
        """Record only submission data already committed to a shared file."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            history = _history_row(connection, query_id)
            file = _file_row(connection, file_name)
            if submission_line not in file["content"].splitlines():
                raise ValueError("Submission line is not present in the file")

            file_names = _array(history["submission_file_names_json"])
            submitted = _array(history["submitted_frame_ids_json"])
            _append_unique(file_names, [file_name])
            _append_unique(submitted, frame_ids)
            connection.execute(
                """
                UPDATE query_history
                SET submission_file_names_json = ?, submitted_frame_ids_json = ?
                WHERE query_id = ?
                """,
                (_json(file_names), _json(submitted), query_id),
            )
            history = _history_row(connection, query_id)

        logger.info(
            "Submission recorded query_id=%s file=%s frames=%d",
            query_id,
            file_name,
            len(frame_ids),
        )
        return _history_record(history)


    def get_recent_history(self, user_id: str) -> list[QueryHistoryRecord]:
        """Return at most the newest twenty query snapshots for one user."""

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


    def list_submission_files(self) -> list[SubmissionFile]:
        """Return every shared submission file for initial hydration."""

        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM submission_files ORDER BY name"
            ).fetchall()
        return [_submission_file(row) for row in rows]


    def list_database_tables(self) -> list[DatabaseTable]:
        """Describe visible tables without exposing internal SQLite tables."""

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
            raise KeyError(
                f"Database table {table_name!r} is not available"
            ) from None

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


    def execute_query(
        self,
        query: str,
        *,
        max_rows: int = 100,
    ) -> DatabaseQueryResponse:
        """Execute an arbitrary raw SQL query against workspace SQLite."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("SQL query cannot be empty")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")

        start_time = time.perf_counter()
        try:
            with self._connection() as connection:
                cursor = connection.execute(cleaned_query)
                if cursor.description is not None:
                    columns = [col[0] for col in cursor.description]
                    fetched_rows = cursor.fetchmany(max_rows)
                    rows = [dict(row) for row in fetched_rows]
                    is_mutation = False
                    rows_affected = 0
                else:
                    columns = []
                    rows = []
                    is_mutation = True
                    rows_affected = cursor.rowcount if cursor.rowcount >= 0 else 0
        except sqlite3.Error as error:
            raise ValueError(f"SQLite execution failed: {error}") from error

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 3)

        return DatabaseQueryResponse(
            query=cleaned_query,
            columns=columns,
            rows=rows,
            rows_affected=rows_affected,
            execution_time_ms=execution_time_ms,
            is_mutation=is_mutation,
        )


    def create_submission_file(self, name: str, content: str) -> SubmissionFile:
        """Create one globally named submission file at revision one."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO submission_files (name, content, is_validated, revision)
                VALUES (?, ?, 0, 1)
                """,
                (name, content),
            )
            file = _submission_file(_file_row(connection, name))
        logger.info("Submission file created name=%s revision=1", name)
        return file


    def update_submission_file(
        self,
        name: str,
        content: str,
        expected_revision: int,
    ) -> SubmissionFile:
        """Replace file content and clear validation when revision matches."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _matching_file(connection, name, expected_revision)
            revision = row["revision"] + 1
            connection.execute(
                """
                UPDATE submission_files
                SET content = ?, is_validated = 0, revision = ?
                WHERE name = ?
                """,
                (content, revision, name),
            )
            file = _submission_file(_file_row(connection, name))
        logger.info("Submission file updated name=%s revision=%d", name, revision)
        return file


    def validate_submission_file(
        self,
        name: str,
        is_validated: bool,
        expected_revision: int,
    ) -> SubmissionFile:
        """Change validation state without allowing an empty validated file."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _matching_file(connection, name, expected_revision)
            if is_validated and not row["content"].strip():
                raise ValueError("Empty submission files cannot be validated")
            revision = row["revision"] + 1
            connection.execute(
                """
                UPDATE submission_files
                SET is_validated = ?, revision = ?
                WHERE name = ?
                """,
                (int(is_validated), revision, name),
            )
            file = _submission_file(_file_row(connection, name))
        logger.info("Submission file validated name=%s revision=%d", name, revision)
        return file


    def delete_submission_file(self, name: str, expected_revision: int) -> None:
        """Delete one shared file when the caller has its latest revision."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            _matching_file(connection, name, expected_revision)
            connection.execute("DELETE FROM submission_files WHERE name = ?", (name,))
        logger.info("Submission file deleted name=%s", name)


    def clear_submission_files(self) -> list[str]:
        """Delete all shared submission files and return their deleted names."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT name FROM submission_files").fetchall()
            names = [str(row["name"]) for row in rows]
            connection.execute("DELETE FROM submission_files")
        logger.info("All submission files cleared count=%d", len(names))
        return names


    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open one short SQLite transaction for an operation."""

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


    def _initialize(self) -> None:
        """Create the Workspace schema and enable concurrent readers."""

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    query_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    result_snapshot_json TEXT NOT NULL,
                    submission_file_names_json TEXT NOT NULL DEFAULT '[]',
                    viewed_frame_ids_json TEXT NOT NULL DEFAULT '[]',
                    submitted_frame_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS query_history_user_created_idx
                ON query_history (user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS submission_files (
                    name TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    is_validated INTEGER NOT NULL DEFAULT 0
                        CHECK (is_validated IN (0, 1)),
                    revision INTEGER NOT NULL CHECK (revision > 0)
                );
                """
            )
        logger.info("Workspace database ready path=%s", self.database_path)


def _utc_now() -> str:
    """Return a sortable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: object) -> str:
    """Encode compact UTF-8 JSON for SQLite."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _array(value: str) -> list[str]:
    """Decode one stored JSON string array."""

    return list(json.loads(value))


def _append_unique(target: list[str], values: list[str]) -> bool:
    """Append unseen values while preserving their first-seen order."""

    changed = False
    for value in values:
        if value not in target:
            target.append(value)
            changed = True
    return changed


def _history_row(connection: sqlite3.Connection, query_id: str) -> sqlite3.Row:
    """Load one history row or report its missing ID."""

    row = connection.execute(
        "SELECT * FROM query_history WHERE query_id = ?", (query_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Query history {query_id!r} not found")
    return row


def _file_row(connection: sqlite3.Connection, name: str) -> sqlite3.Row:
    """Load one shared file row or report its missing name."""

    row = connection.execute(
        "SELECT * FROM submission_files WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Submission file {name!r} not found")
    return row


def _matching_file(
    connection: sqlite3.Connection,
    name: str,
    expected_revision: int,
) -> sqlite3.Row:
    """Load a file and reject writes based on a stale revision."""

    row = _file_row(connection, name)
    if row["revision"] != expected_revision:
        raise RevisionConflict(_submission_file(row))
    return row


def _history_record(row: sqlite3.Row) -> QueryHistoryRecord:
    """Convert a SQLite history row into its public contract."""

    return QueryHistoryRecord(
        query_id=row["query_id"],
        query_text=row["query_text"],
        submission_files=_array(row["submission_file_names_json"]),
        result_snapshot=json.loads(row["result_snapshot_json"]),
        frame_activity=FrameActivity(
            viewed_frame_ids=_array(row["viewed_frame_ids_json"]),
            submitted_frame_ids=_array(row["submitted_frame_ids_json"]),
        ),
    )


def _submission_file(row: sqlite3.Row) -> SubmissionFile:
    """Convert a SQLite file row into its public contract."""

    return SubmissionFile(
        name=row["name"],
        content=row["content"],
        is_validated=bool(row["is_validated"]),
        revision=row["revision"],
    )


def _table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[DatabaseColumn]:
    """Project SQLite schema facts for one already-allowlisted table name."""

    return [
        DatabaseColumn(
            name=row["name"],
            type=row["type"],
            nullable=not bool(row["notnull"]) and not bool(row["pk"]),
            primary_key=bool(row["pk"]),
        )
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    ]


__all__ = ["RevisionConflict", "WorkspaceStore"]
