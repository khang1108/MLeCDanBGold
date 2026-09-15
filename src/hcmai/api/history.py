"""Persist query replay history.

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

from hcmai.api.contracts.history import (
    FrameActivity,
    QueryHistoryCreate,
    QueryHistoryRecord,
    QueryInteractionEventCreate,
    QueryInteractionEventRecord,
    QueryOperationMetadata,
)
from hcmai.common.utils.logging import get_logger


logger = get_logger(__name__)
_DATABASE_VERSION = 4

_RETIRED_ANSWER_TABLES = frozenset(
    {
        "answer_workspace_state",
        "answer_candidates",
        "submission_attempts",
        "submission_files",
    }
)


class WorkspaceStore:
    """Persist replay snapshots, viewed-frame activity, and history metadata."""

    def __init__(self, database_path: str | Path) -> None:
        """Create the database directory and migrate older schemas to v4."""

        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_history(self, data: QueryHistoryCreate) -> QueryHistoryRecord:
        """Create one replay snapshot with empty viewed-frame activity."""

        op_meta_json = (
            _json(data.operation_metadata.model_dump())
            if data.operation_metadata is not None
            else "{}"
        )

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO query_history (
                    query_id, user_id, query_text, result_snapshot_json,
                    viewed_frame_ids_json, operation_metadata_json, created_at
                ) VALUES (?, ?, ?, ?, '[]', ?, ?)
                """,
                (
                    data.query_id,
                    data.user_id,
                    data.query_text,
                    _json(data.result_snapshot),
                    op_meta_json,
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

    def record_interaction_event(
        self,
        query_id: str,
        event: QueryInteractionEventCreate,
    ) -> QueryInteractionEventRecord:
        """Record one append-only interaction event for an existing query."""

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Verify query exists
            row = connection.execute(
                "SELECT query_id FROM query_history WHERE query_id = ?",
                (query_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Query history {query_id!r} not found")

            # Monotonically increasing sequence_id per query
            seq_row = connection.execute(
                "SELECT COALESCE(MAX(sequence_id), 0) + 1 FROM query_interaction_events WHERE query_id = ?",
                (query_id,),
            ).fetchone()
            sequence_id = int(seq_row[0])
            created_at = _utc_now()

            connection.execute(
                """
                INSERT INTO query_interaction_events (
                    query_id, sequence_id, event_type, semantic_revision,
                    event_id, frame_id, video_id, timestamp_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    sequence_id,
                    event.event_type,
                    event.semantic_revision,
                    event.event_id,
                    event.frame_id,
                    event.video_id,
                    event.timestamp_ms,
                    created_at,
                ),
            )

        logger.info(
            "Interaction event recorded query_id=%s sequence_id=%d event_type=%s",
            query_id,
            sequence_id,
            event.event_type,
        )
        return QueryInteractionEventRecord(
            query_id=query_id,
            sequence_id=sequence_id,
            event_type=event.event_type,
            semantic_revision=event.semantic_revision,
            event_id=event.event_id,
            frame_id=event.frame_id,
            video_id=event.video_id,
            timestamp_ms=event.timestamp_ms,
            created_at=created_at,
        )

    def get_interaction_events(
        self,
        query_id: str,
    ) -> list[QueryInteractionEventRecord]:
        """Return all interaction events for a query in sequence order."""

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM query_interaction_events
                WHERE query_id = ?
                ORDER BY sequence_id ASC
                """,
                (query_id,),
            ).fetchall()

        return [
            QueryInteractionEventRecord(
                query_id=row["query_id"],
                sequence_id=row["sequence_id"],
                event_type=row["event_type"],
                semantic_revision=row["semantic_revision"],
                event_id=row["event_id"],
                frame_id=row["frame_id"],
                video_id=row["video_id"],
                timestamp_ms=row["timestamp_ms"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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

    def _initialize(self) -> None:
        """Atomically preserve query history and migrate to v4."""

        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _DATABASE_VERSION:
                raise RuntimeError(
                    f"Workspace database version {version} is newer than supported "
                    f"{_DATABASE_VERSION}"
                )

            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "query_history" not in tables:
                self._create_query_history_table(connection)
            else:
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
                          viewed_frame_ids_json, operation_metadata_json, created_at
                        )
                        SELECT query_id, user_id, query_text, result_snapshot_json,
                          viewed_frame_ids_json, '{}', created_at
                        FROM query_history_legacy
                        """
                    )
                    connection.execute("DROP TABLE query_history_legacy")
                elif "operation_metadata_json" not in columns:
                    connection.execute(
                        "ALTER TABLE query_history ADD COLUMN operation_metadata_json TEXT NOT NULL DEFAULT '{}'"
                    )

            self._create_interaction_events_table(connection)

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
        """Create query replay storage with operation metadata column."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_history (
              query_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              query_text TEXT NOT NULL,
              result_snapshot_json TEXT NOT NULL,
              viewed_frame_ids_json TEXT NOT NULL DEFAULT '[]',
              operation_metadata_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _create_interaction_events_table(connection: sqlite3.Connection) -> None:
        """Create append-only interaction events table."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS query_interaction_events (
              query_id TEXT NOT NULL,
              sequence_id INTEGER NOT NULL,
              event_type TEXT NOT NULL,
              semantic_revision INTEGER NOT NULL,
              event_id TEXT,
              frame_id TEXT,
              video_id TEXT,
              timestamp_ms INTEGER,
              created_at TEXT NOT NULL,
              PRIMARY KEY (query_id, sequence_id),
              FOREIGN KEY (query_id) REFERENCES query_history (query_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS query_interaction_query_idx
            ON query_interaction_events (query_id, sequence_id ASC)
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

    op_meta = None
    if "operation_metadata_json" in row.keys() and row["operation_metadata_json"]:
        try:
            raw_meta = json.loads(row["operation_metadata_json"])
            if raw_meta and isinstance(raw_meta, dict):
                op_meta = QueryOperationMetadata.model_validate(raw_meta)
        except Exception:
            op_meta = None

    return QueryHistoryRecord(
        query_id=row["query_id"],
        query_text=row["query_text"],
        result_snapshot=json.loads(row["result_snapshot_json"]),
        frame_activity=FrameActivity(
            viewed_frame_ids=_array(row["viewed_frame_ids_json"]),
        ),
        operation_metadata=op_meta,
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


__all__ = ["WorkspaceStore", "_DATABASE_VERSION"]
