"""Tests for query history operation metadata and interaction event logging."""

from __future__ import annotations

import sqlite3
import pytest
from fastapi.testclient import TestClient

from hcmai.api.contracts.history import (
    QueryHistoryCreate,
    QueryHistoryRecord,
    QueryInteractionEventCreate,
    QueryInteractionEventRecord,
    QueryOperationMetadata,
)
from hcmai.api.history import WorkspaceStore, _DATABASE_VERSION
from hcmai.app import create_app


def _sample_snapshot() -> dict:
    return {
        "results": [
            {
                "frame_id": "frame-1",
                "video_id": "V01",
                "frame_idx": 100,
                "timestamp_ms": 4000,
                "score": 0.95,
            }
        ],
        "latency": {"total_ms": 12.5},
    }


class DummySearchService:
    """Minimal search service stub satisfying frame validation."""

    def __init__(self) -> None:
        self.frames = {
            "frame-1": type(
                "Frame",
                (),
                {"frame_id": "frame-1", "video_id": "V01", "frame_idx": 100, "timestamp_ms": 4000},
            )(),
        }

    def get_frame(self, frame_id: str):
        if frame_id in self.frames:
            return self.frames[frame_id]
        raise KeyError(f"Frame {frame_id} not found")


def test_workspace_store_migrates_v3_to_v4(tmp_path) -> None:
    """Test migration from v3 schema to v4 adds operation_metadata and interaction table."""
    db_path = tmp_path / "workspace_test.db"

    # Create v3 database manually
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE query_history (
          query_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          query_text TEXT NOT NULL,
          result_snapshot_json TEXT NOT NULL,
          viewed_frame_ids_json TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO query_history VALUES (
          'q-legacy', 'user-1', 'red boat', '{"results": []}', '[]', '2026-09-15T00:00:00Z'
        )
        """
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    # Open with WorkspaceStore
    store = WorkspaceStore(db_path)
    records = store.get_recent_history("user-1")
    assert len(records) == 1
    assert records[0].query_id == "q-legacy"
    assert records[0].operation_metadata is None

    # Verify DB version and table existence
    conn = sqlite3.connect(db_path)
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    assert version == 4
    cols = {row[1] for row in conn.execute("PRAGMA table_info(query_history)").fetchall()}
    assert "operation_metadata_json" in cols

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "query_interaction_events" in tables
    conn.close()


def test_workspace_store_roundtrip_operation_metadata(tmp_path) -> None:
    """Test saving and retrieving query history with operation metadata."""
    store = WorkspaceStore(tmp_path / "store.db")
    op_meta = QueryOperationMetadata(
        semantic_revision=2,
        operation_kind="patch_events",
        affected_event_ids=["E2"],
        image_added=["ast_1"],
        image_removed=["ast_old"],
        search_only=False,
    )
    req = QueryHistoryCreate(
        query_id="q-meta-1",
        user_id="user-1",
        query_text="E2: red boat",
        result_snapshot=_sample_snapshot(),
        operation_metadata=op_meta,
    )
    record = store.create_history(req)
    assert record.operation_metadata is not None
    assert record.operation_metadata.semantic_revision == 2
    assert record.operation_metadata.operation_kind == "patch_events"
    assert record.operation_metadata.affected_event_ids == ["E2"]
    assert record.operation_metadata.image_added == ["ast_1"]
    assert record.operation_metadata.image_removed == ["ast_old"]
    assert record.operation_metadata.search_only is False

    loaded = store.get_recent_history("user-1")
    assert len(loaded) == 1
    assert loaded[0].operation_metadata == op_meta


def test_workspace_store_roundtrip_feedback_operation_metadata(tmp_path) -> None:
    """Test saving and retrieving query history with chat feedback metadata."""
    store = WorkspaceStore(tmp_path / "store.db")
    op_meta = QueryOperationMetadata(
        semantic_revision=1,
        operation_kind="chat_feedback",
        affected_event_ids=["E1"],
        action="refine",
        scope="all_videos",
        feedback_revision=1,
    )
    req = QueryHistoryCreate(
        query_id="q-feedback-1",
        user_id="user-1",
        query_text="focus on red shirt",
        result_snapshot=_sample_snapshot(),
        operation_metadata=op_meta,
    )
    record = store.create_history(req)
    assert record.operation_metadata is not None
    assert record.operation_metadata.action == "refine"
    assert record.operation_metadata.scope == "all_videos"
    assert record.operation_metadata.feedback_revision == 1

    loaded = store.get_recent_history("user-1")
    assert len(loaded) == 1
    assert loaded[0].operation_metadata == op_meta


def test_workspace_store_interaction_events_monotonically_ordered(tmp_path) -> None:
    """Test append-only interaction events preserve sequence order per query."""
    store = WorkspaceStore(tmp_path / "store.db")
    req = QueryHistoryCreate(
        query_id="q-events-1",
        user_id="user-1",
        query_text="red boat",
        result_snapshot=_sample_snapshot(),
    )
    store.create_history(req)

    e1 = store.record_interaction_event(
        "q-events-1",
        QueryInteractionEventCreate(
            event_type="result_open",
            semantic_revision=1,
            event_id="E1",
            frame_id="frame-1",
            video_id="V01",
            timestamp_ms=4000,
        ),
    )
    assert e1.sequence_id == 1
    assert e1.event_type == "result_open"

    e2 = store.record_interaction_event(
        "q-events-1",
        QueryInteractionEventCreate(
            event_type="submission",
            semantic_revision=1,
            frame_id="frame-1",
            video_id="V01",
            timestamp_ms=4000,
        ),
    )
    assert e2.sequence_id == 2
    assert e2.event_type == "submission"

    events = store.get_interaction_events("q-events-1")
    assert len(events) == 2
    assert events[0].sequence_id == 1
    assert events[1].sequence_id == 2


def test_interaction_events_missing_query_id(tmp_path) -> None:
    """Recording an event on a nonexistent query raises KeyError."""
    store = WorkspaceStore(tmp_path / "store.db")
    with pytest.raises(KeyError):
        store.record_interaction_event(
            "nonexistent-q",
            QueryInteractionEventCreate(
                event_type="result_open",
                semantic_revision=0,
            ),
        )


def test_history_routes_interaction_events_api(tmp_path) -> None:
    """HTTP API tests for query-history events endpoint."""
    store = WorkspaceStore(tmp_path / "api_test.db")
    search_service = DummySearchService()
    app = create_app(search_service=search_service, workspace_store=store)
    client = TestClient(app)

    # 1. Create history
    create_res = client.post(
        "/api/v1/query-history",
        json={
            "query_id": "q-http-1",
            "user_id": "team-a",
            "query_text": "sample",
            "result_snapshot": _sample_snapshot(),
            "operation_metadata": {
                "semantic_revision": 1,
                "operation_kind": "initial_resolve",
                "affected_event_ids": ["E1"],
                "image_added": [],
                "image_removed": [],
                "search_only": False,
            },
        },
    )
    assert create_res.status_code == 201, create_res.text
    assert create_res.json()["operation_metadata"]["semantic_revision"] == 1

    # 2. Add interaction event
    event_res = client.post(
        "/api/v1/query-history/q-http-1/events",
        json={
            "event_type": "result_open",
            "semantic_revision": 1,
            "event_id": "E1",
            "frame_id": "frame-1",
            "video_id": "V01",
            "timestamp_ms": 4000,
        },
    )
    assert event_res.status_code == 201, event_res.text
    data = event_res.json()
    assert data["sequence_id"] == 1
    assert data["event_type"] == "result_open"
    assert data["semantic_revision"] == 1

    # 3. Missing query returns 404
    missing_res = client.post(
        "/api/v1/query-history/q-missing/events",
        json={
            "event_type": "result_open",
            "semantic_revision": 1,
        },
    )
    assert missing_res.status_code == 404

    # 4. Invalid event payload returns 422
    invalid_res = client.post(
        "/api/v1/query-history/q-http-1/events",
        json={
            "event_type": "invalid_kind",
            "semantic_revision": -1,
        },
    )
    assert invalid_res.status_code == 422
