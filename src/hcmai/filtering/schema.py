"""Define the immutable SQLite schema shared by build and serving paths.

This module owns schema shape and indexes only. Artifact population belongs to
the offline builder, while read-only lifecycle validation belongs to catalog.
"""

from __future__ import annotations

import sqlite3


CATALOG_SCHEMA_VERSION = "filter-catalog-v1"

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "catalog_metadata": frozenset(
        {
            "id",
            "schema_version",
            "catalog_version",
            "built_at",
            "frame_count",
            "source_lineage_json",
            "title_available",
            "caption_available",
            "ocr_available",
            "objects_available",
            "asr_available",
        }
    ),
    "frames": frozenset(
        {
            "frame_id",
            "video_id",
            "frame_idx",
            "timestamp_ms",
            "folder_id",
            "title",
            "title_norm",
            "caption",
            "caption_norm",
            "ocr",
            "ocr_norm",
            "asr",
            "asr_norm",
        }
    ),
    "frame_objects": frozenset(
        {"frame_id", "label_norm", "object_count"}
    ),
}

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE catalog_metadata (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        schema_version TEXT NOT NULL,
        catalog_version TEXT NOT NULL,
        built_at TEXT NOT NULL,
        frame_count INTEGER NOT NULL CHECK (frame_count >= 0),
        source_lineage_json TEXT NOT NULL,
        title_available INTEGER NOT NULL CHECK (title_available IN (0, 1)),
        caption_available INTEGER NOT NULL CHECK (caption_available IN (0, 1)),
        ocr_available INTEGER NOT NULL CHECK (ocr_available IN (0, 1)),
        objects_available INTEGER NOT NULL CHECK (objects_available IN (0, 1)),
        asr_available INTEGER NOT NULL CHECK (asr_available IN (0, 1))
    )
    """,
    """
    CREATE TABLE frames (
        frame_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        frame_idx INTEGER NOT NULL CHECK (frame_idx >= 0),
        timestamp_ms INTEGER NOT NULL CHECK (timestamp_ms >= 0),
        folder_id TEXT NOT NULL,
        title TEXT,
        title_norm TEXT,
        caption TEXT,
        caption_norm TEXT,
        ocr TEXT,
        ocr_norm TEXT,
        asr TEXT,
        asr_norm TEXT
    )
    """,
    """
    CREATE TABLE frame_objects (
        frame_id TEXT NOT NULL,
        label_norm TEXT NOT NULL CHECK (length(label_norm) > 0),
        object_count INTEGER NOT NULL CHECK (object_count >= 0),
        PRIMARY KEY (frame_id, label_norm),
        FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
    )
    """,
    """
    CREATE INDEX idx_frames_order
    ON frames(video_id, timestamp_ms, frame_idx, frame_id)
    """,
    """
    CREATE INDEX idx_frames_folder_order
    ON frames(folder_id, video_id, timestamp_ms, frame_idx, frame_id)
    """,
    """
    CREATE INDEX idx_frame_objects_match
    ON frame_objects(label_norm, object_count, frame_id)
    """,
)


def create_catalog_schema(connection: sqlite3.Connection) -> None:
    """Create an empty Filter catalog schema on one writable connection."""

    connection.execute("PRAGMA foreign_keys=ON")
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)

