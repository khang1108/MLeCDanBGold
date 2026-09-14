"""Persist query replay history and expose safe SQLite inspection helpers.

This module owns query-history schema migration and read/write transactions. It
does not persist answer candidates or DRES submission outcomes.
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
)
from hcmai.common.utils.logging import get_logger


logger = get_logger(__name__)
_DATABASE_VERSION = 3
_DATABASE_TABLE_ORDER = {"query_history": "created_at DESC, rowid DESC"}
_RETIRED_ANSWER_TABLES = frozenset(
    {
        "answer_workspace_state",
        "answer_candidates",
        "submission_attempts",
        "submission_files",
    }
)
_RETIRED_ANSWER_OBJECTS = frozenset({"answer_frame_unique"})
_SQLITE_DML_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
    }
)
_SQLITE_TABLE_TARGET_SECOND_ARGUMENT_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    }
)
_SQLITE_RETIRED_TABLE_ACTIONS = _SQLITE_DML_ACTIONS | frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    }
)


class WorkspaceStore:
    """Persist replay snapshots, viewed-frame activity, and history metadata."""

    def __init__(self, database_path: str | Path) -> None:
        """Create the database directory and migrate older schemas to v3."""

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
            _append_unique(frame_ids, [frame_id])
            connection.execute(
                "UPDATE query_history SET viewed_frame_ids_json = ? WHERE query_id = ?",
                (_json(frame_ids), query_id),
            )
            row = _history_row(connection, query_id)
        logger.info("Viewed frame recorded query_id=%s frame_id=%s", query_id, frame_id)
        return _history_record(row)

    def get_recent_history(self, user_id: str) -> list[QueryHistoryRecord]:
        """Return at most the newest twenty snapshots belonging to one user."""

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

    def list_database_tables(self) -> list[DatabaseTable]:
        """Describe the sole application table exposed to database browsing."""

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
        """Read one stable page from the allowlisted query-history table."""

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

    def execute_query(
        self,
        query: str,
        *,
        max_rows: int = 100,
    ) -> DatabaseQueryResponse:
        """Run one database-console statement while blocking retired answer tables."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("SQL query cannot be empty")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")

        start_time = time.perf_counter()
        denied_objects: list[str] = []

        def authorize_retired_answer_storage(
            action: int,
            table_name: str | None,
            second_argument: str | None,
            _database_name: str | None,
            _trigger_name: str | None,
        ) -> int:
            """Prevent SQL-console writes from reviving the removed answer store."""

            target_table = (
                second_argument
                if action in _SQLITE_TABLE_TARGET_SECOND_ARGUMENT_ACTIONS
                else table_name
            )
            if (
                action in _SQLITE_RETIRED_TABLE_ACTIONS
                and target_table is not None
                and target_table.casefold() in _RETIRED_ANSWER_TABLES
            ):
                denied_objects.append(target_table)
                return sqlite3.SQLITE_DENY
            if (
                action
                in {
                    sqlite3.SQLITE_DROP_INDEX,
                    sqlite3.SQLITE_DROP_TEMP_INDEX,
                    sqlite3.SQLITE_DROP_TRIGGER,
                    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
                }
                and table_name is not None
                and table_name.casefold() in _RETIRED_ANSWER_OBJECTS
            ):
                denied_objects.append(table_name)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        try:
            with self._connection() as connection:
                connection.set_authorizer(authorize_retired_answer_storage)
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
            if denied_objects:
                raise ValueError(
                    "Retired answer storage tables cannot be accessed or recreated"
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
        """Open one short SQLite transaction and roll back failed operations."""

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
        """Atomically preserve query history and retire all answer-only tables."""

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _DATABASE_VERSION:
                raise RuntimeError(
                    f"Workspace database version {version} is newer than supported "
                    f"{_DATABASE_VERSION}"
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
                        for row in connection.execute(
                            "PRAGMA table_info(query_history)"
                        ).fetchall()
                    }
                    if {
                        "submission_file_names_json",
                        "submitted_frame_ids_json",
                    } & columns:
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
            else:
                self._create_query_history_table(connection)

            for table_name in sorted(_RETIRED_ANSWER_TABLES):
                connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS query_history_user_created_idx
                ON query_history (user_id, created_at DESC)
                """
            )
            connection.execute(f"PRAGMA user_version = {_DATABASE_VERSION}")
            logger.info(
                "Query history database ready path=%s version=%d",
                self.database_path,
                _DATABASE_VERSION,
            )

    @staticmethod
    def _create_query_history_table(connection: sqlite3.Connection) -> None:
        """Create query replay storage without answer or submission columns."""

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
    """Convert a SQLite replay row into its public contract."""

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
    """Project schema facts for an allowlisted table name."""

    return [
        DatabaseColumn(
            name=row["name"],
            type=row["type"],
            nullable=not bool(row["notnull"]) and not bool(row["pk"]),
            primary_key=bool(row["pk"]),
        )
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    ]


__all__ = ["WorkspaceStore"]
