"""Open and pool validated Filter SQLite catalogs for read-only serving.

The catalog bounds connection/page-cache memory and never creates or repairs
artifacts. Offline publication is intentionally outside this module.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schema import CATALOG_SCHEMA_VERSION, REQUIRED_COLUMNS


class FilterCatalogUnavailableError(RuntimeError):
    """Raised when the optional Filter catalog cannot be opened or borrowed."""


class FilterCatalogCorruptError(RuntimeError):
    """Raised when an opened catalog violates its published invariants."""


@dataclass(frozen=True)
class CatalogAvailability:
    """Global modality availability recorded at offline build time."""

    title: bool
    caption: bool
    ocr: bool
    objects: bool
    asr: bool


@dataclass(frozen=True)
class FilterCatalogInfo:
    """Validated catalog facts safe to expose through application health."""

    schema_version: str
    catalog_version: str
    built_at: str
    frame_count: int
    source_lineage: dict[str, Any]
    availability: CatalogAvailability


class FilterCatalog:
    """A fixed-size pool of read-only connections to one validated catalog."""

    def __init__(
        self,
        *,
        path: Path,
        connections: list[sqlite3.Connection],
        info: FilterCatalogInfo,
    ) -> None:
        """Own already-configured connections after successful validation."""

        self._path = path
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(
            maxsize=len(connections)
        )
        for connection in connections:
            self._pool.put_nowait(connection)
        self._pool_size = len(connections)
        self._created_connection_count = len(connections)
        self._closed = False
        self._state_lock = threading.Lock()
        self.info = info

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        pool_size: int = 4,
        cache_kib: int = 8192,
    ) -> FilterCatalog:
        """Open, configure, and validate a bounded read-only catalog pool."""

        catalog_path = Path(path).expanduser().resolve()
        if pool_size < 1 or pool_size > 4:
            raise ValueError("pool_size must be between 1 and 4")
        if cache_kib < 1:
            raise ValueError("cache_kib must be positive")
        if not catalog_path.is_file():
            raise FilterCatalogUnavailableError(
                f"Filter catalog is not available at {catalog_path}"
            )

        connections: list[sqlite3.Connection] = []
        try:
            for _ in range(pool_size):
                connections.append(
                    _open_read_only_connection(catalog_path, cache_kib=cache_kib)
                )
            info = _validate_catalog(connections[0])
        except FilterCatalogCorruptError:
            for connection in connections:
                connection.close()
            raise
        except (OSError, sqlite3.Error) as error:
            for connection in connections:
                connection.close()
            raise FilterCatalogUnavailableError(
                f"Could not open Filter catalog: {type(error).__name__}: {error}"
            ) from error

        return cls(path=catalog_path, connections=connections, info=info)

    @property
    def pool_size(self) -> int:
        """Return the configured maximum concurrent SQLite connections."""

        return self._pool_size

    @property
    def created_connection_count(self) -> int:
        """Return the fixed connection count for resource-bound assertions."""

        return self._created_connection_count

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Borrow one connection, waiting rather than growing the pool."""

        with self._state_lock:
            if self._closed:
                raise FilterCatalogUnavailableError("Filter catalog is closed")
        connection = self._pool.get()
        try:
            yield connection
        finally:
            with self._state_lock:
                closed = self._closed
            if closed:
                connection.close()
            else:
                self._pool.put(connection)

    def close(self) -> None:
        """Close every idle pooled connection exactly once."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        while True:
            try:
                connection = self._pool.get_nowait()
            except queue.Empty:
                break
            connection.close()


def _open_read_only_connection(
    path: Path,
    *,
    cache_kib: int,
) -> sqlite3.Connection:
    """Open one SQLite connection with the V1 runtime memory constraints."""

    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute(f"PRAGMA cache_size=-{cache_kib}")
    return connection


def _validate_catalog(connection: sqlite3.Connection) -> FilterCatalogInfo:
    """Validate required schema, metadata, lineage JSON, and frame count."""

    try:
        for table, required in REQUIRED_COLUMNS.items():
            columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            missing = required - columns
            if missing:
                raise FilterCatalogCorruptError(
                    f"Filter catalog table {table} is missing columns: "
                    f"{', '.join(sorted(missing))}"
                )

        rows = connection.execute("SELECT * FROM catalog_metadata").fetchall()
        if len(rows) != 1 or rows[0]["id"] != 1:
            raise FilterCatalogCorruptError(
                "Filter catalog must contain one metadata row with id=1"
            )
        metadata = rows[0]
        if metadata["schema_version"] != CATALOG_SCHEMA_VERSION:
            raise FilterCatalogCorruptError(
                "Filter catalog schema_version mismatch: "
                f"expected {CATALOG_SCHEMA_VERSION}, "
                f"got {metadata['schema_version']}"
            )

        actual_count = connection.execute(
            "SELECT COUNT(*) AS count FROM frames"
        ).fetchone()["count"]
        if actual_count != metadata["frame_count"]:
            raise FilterCatalogCorruptError(
                "Filter catalog frame_count does not match frames table"
            )
        try:
            lineage = json.loads(metadata["source_lineage_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise FilterCatalogCorruptError(
                "Filter catalog source_lineage_json is invalid"
            ) from error
        if not isinstance(lineage, dict):
            raise FilterCatalogCorruptError(
                "Filter catalog source_lineage_json must be an object"
            )

        return FilterCatalogInfo(
            schema_version=metadata["schema_version"],
            catalog_version=metadata["catalog_version"],
            built_at=metadata["built_at"],
            frame_count=metadata["frame_count"],
            source_lineage=lineage,
            availability=CatalogAvailability(
                title=bool(metadata["title_available"]),
                caption=bool(metadata["caption_available"]),
                ocr=bool(metadata["ocr_available"]),
                objects=bool(metadata["objects_available"]),
                asr=bool(metadata["asr_available"]),
            ),
        )
    except FilterCatalogCorruptError:
        raise
    except sqlite3.Error as error:
        raise FilterCatalogCorruptError(
            f"Filter catalog validation failed: {type(error).__name__}: {error}"
        ) from error
