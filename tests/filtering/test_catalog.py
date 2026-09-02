"""Tests for the bounded read-only Filter catalog runtime boundary."""

from __future__ import annotations

import sqlite3
import threading

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hcmai.filtering.catalog import (
    FilterCatalog,
    FilterCatalogCorruptError,
    FilterCatalogUnavailableError,
)
from hcmai.filtering.schema import CATALOG_SCHEMA_VERSION, create_catalog_schema


def _write_catalog(
    path: Path,
    *,
    schema_version: str = CATALOG_SCHEMA_VERSION,
) -> None:
    """Create a two-frame catalog whose expected metadata is easy to inspect."""

    connection = sqlite3.connect(path)
    create_catalog_schema(connection)
    connection.execute(
        """
        INSERT INTO catalog_metadata (
            id, schema_version, catalog_version, built_at, frame_count,
            source_lineage_json, title_available, caption_available,
            ocr_available, objects_available, asr_available
        ) VALUES (1, ?, 'fixture-v1', '2026-09-02T00:00:00Z', 2, '{}', 1, 1, 1, 1, 0)
        """,
        (schema_version,),
    )
    connection.executemany(
        """
        INSERT INTO frames (
            frame_id, video_id, frame_idx, timestamp_ms, folder_id,
            title, title_norm, caption, caption_norm, ocr, ocr_norm, asr, asr_norm
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "L21_V001_000001", "L21_V001", 25, 1000, "L21",
                "Title", "title", "Caption", "caption", None, None, None, None,
            ),
            (
                "L21_V001_000002", "L21_V001", 50, 2000, "L21",
                "Title", "title", None, None, "OCR", "ocr", None, None,
            ),
        ],
    )
    connection.execute(
        "INSERT INTO frame_objects(frame_id, label_norm, object_count) VALUES (?, ?, ?)",
        ("L21_V001_000001", "person", 3),
    )
    connection.commit()
    connection.close()


def test_open_validates_metadata_and_configures_read_only_connections(
    tmp_path: Path,
) -> None:
    """Expose validated catalog facts while prohibiting runtime writes."""

    path = tmp_path / "filter.sqlite"
    _write_catalog(path)

    catalog = FilterCatalog.open(path, pool_size=2, cache_kib=4096)

    assert catalog.info.catalog_version == "fixture-v1"
    assert catalog.info.frame_count == 2
    assert catalog.info.availability.caption is True
    assert catalog.info.availability.asr is False
    assert catalog.pool_size == 2
    with catalog.connection() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert connection.execute("PRAGMA cache_size").fetchone()[0] == -4096
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM frames")

    catalog.close()


def test_open_rejects_missing_and_wrong_schema_catalogs(tmp_path: Path) -> None:
    """Distinguish unavailable deployment state from corrupt catalog state."""

    with pytest.raises(FilterCatalogUnavailableError):
        FilterCatalog.open(tmp_path / "missing.sqlite")

    wrong_schema = tmp_path / "wrong.sqlite"
    _write_catalog(wrong_schema, schema_version="filter-catalog-v0")
    with pytest.raises(FilterCatalogCorruptError, match="schema_version"):
        FilterCatalog.open(wrong_schema)


def test_open_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    """Prevent serving a partially published or externally modified catalog."""

    path = tmp_path / "mismatch.sqlite"
    _write_catalog(path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE catalog_metadata SET frame_count = 3 WHERE id = 1")
    connection.commit()
    connection.close()

    with pytest.raises(FilterCatalogCorruptError, match="frame_count"):
        FilterCatalog.open(path)


def test_pool_blocks_fifth_borrower_until_a_connection_returns(
    tmp_path: Path,
) -> None:
    """Bound concurrent SQLite memory instead of opening overflow connections."""

    path = tmp_path / "bounded.sqlite"
    _write_catalog(path)
    catalog = FilterCatalog.open(path, pool_size=4)
    borrowed = [catalog.connection() for _ in range(4)]
    connections = [manager.__enter__() for manager in borrowed]
    fifth_entered = threading.Event()

    def borrow_fifth() -> int:
        with catalog.connection() as connection:
            fifth_entered.set()
            return connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(borrow_fifth)
        assert fifth_entered.wait(timeout=0.05) is False
        borrowed[0].__exit__(None, None, None)
        assert future.result(timeout=1) == 2

    for manager in borrowed[1:]:
        manager.__exit__(None, None, None)
    assert len(connections) == 4
    assert catalog.created_connection_count == 4
    catalog.close()


def test_close_is_idempotent_and_prevents_new_borrows(tmp_path: Path) -> None:
    """Make application shutdown deterministic without double-close failures."""

    path = tmp_path / "close.sqlite"
    _write_catalog(path)
    catalog = FilterCatalog.open(path)

    catalog.close()
    catalog.close()

    with pytest.raises(FilterCatalogUnavailableError, match="closed"):
        with catalog.connection():
            pass
