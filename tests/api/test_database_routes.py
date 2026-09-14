"""Tests for safe HTTP browsing of the workspace SQLite database."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from hcmai.api.contracts import QueryHistoryCreate
from hcmai.api.history import WorkspaceStore
from hcmai.app import create_app


pytestmark = pytest.mark.usefixtures("inline_router_threadpool")


def _request(app, path: str) -> httpx.Response:
    """Send one GET request through ASGI without starting app lifespan."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(send())


def _post_json(app, path: str, json_data: dict) -> httpx.Response:
    """Send one POST request with JSON body through ASGI without starting app lifespan."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, json=json_data)

    return asyncio.run(send())


@pytest.fixture
def workspace_store(tmp_path):
    """Create a populated temporary SQLite store."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    store.create_history(
        QueryHistoryCreate(
            query_id="query-1",
            user_id="user-1",
            query_text="red bus",
            result_snapshot={"results": []},
        )
    )
    workspace = store.get_answer_workspace("eval-1", "task-1", "KIS task")
    store.add_text_candidate(
        text="answer",
        user_id="user-1",
        evaluation_id="eval-1",
        task_scope_key="task-1",
        task_name="KIS task",
        expected_workspace_revision=workspace.revision,
    )
    return store


@pytest.fixture
def database_app(workspace_store):
    """Create a populated temporary SQLite store behind the production router."""

    return create_app(workspace_store=workspace_store)


def test_workspace_store_executes_select_query(workspace_store) -> None:
    """Execute SELECT query and verify columns, rows, and timing."""

    result = workspace_store.execute_query(
        "SELECT query_id, user_id FROM query_history WHERE query_id = 'query-1'"
    )
    assert result.columns == ["query_id", "user_id"]
    assert len(result.rows) == 1
    assert result.rows[0]["query_id"] == "query-1"
    assert result.is_mutation is False
    assert result.execution_time_ms >= 0.0


def test_workspace_store_executes_unrelated_mutation_query(workspace_store) -> None:
    """Keep non-answer database-console mutations compatible with existing use."""

    result = workspace_store.execute_query(
        "UPDATE query_history SET query_text = 'updated' WHERE query_id = 'query-1'"
    )
    assert result.is_mutation is True
    assert result.rows_affected == 1

    select_result = workspace_store.execute_query(
        "SELECT query_text FROM query_history WHERE query_id = 'query-1'"
    )
    assert select_result.rows == [{"query_text": "updated"}]


def test_workspace_store_raises_on_invalid_syntax(workspace_store) -> None:
    """Invalid syntax raises ValueError with SQLite error detail."""

    with pytest.raises(ValueError, match="syntax error"):
        workspace_store.execute_query("SELCT * FROM query_history")


@pytest.mark.parametrize(
    "query",
    [
        "UPDATE answer_workspace_state SET revision = revision + 1 WHERE singleton_id = 1",
        "UPDATE answer_candidates SET text = 'changed' WHERE kind = 'TEXT'",
        "DELETE FROM submission_attempts",
        "ALTER TABLE answer_candidates ADD COLUMN bypassed_revision INTEGER",
        "DROP TABLE answer_candidates",
        "CREATE INDEX bypassed_candidate_index ON answer_candidates(text)",
        "DROP INDEX answer_frame_unique",
    ],
)
def test_workspace_store_rejects_raw_answer_table_writes(workspace_store, query: str) -> None:
    """Protect typed workspace revisions and durable attempt snapshots from SQL bypasses."""

    with pytest.raises(ValueError, match="Answer workspace tables are managed"):
        workspace_store.execute_query(query)


def test_workspace_store_authorizer_rejects_triggered_answer_table_writes(workspace_store) -> None:
    """Deny protected writes even when a SQL trigger attempts them indirectly."""

    before = workspace_store.execute_query(
        "SELECT revision FROM answer_workspace_state WHERE singleton_id = 1"
    ).rows[0]["revision"]
    workspace_store.execute_query(
        "CREATE TRIGGER bypass_workspace_revision AFTER UPDATE ON query_history "
        "BEGIN UPDATE answer_workspace_state SET revision = revision + 1; END"
    )

    with pytest.raises(ValueError, match="Answer workspace tables are managed"):
        workspace_store.execute_query(
            "UPDATE query_history SET query_text = 'trigger attempt' WHERE query_id = 'query-1'"
        )

    after = workspace_store.execute_query(
        "SELECT revision FROM answer_workspace_state WHERE singleton_id = 1"
    ).rows[0]["revision"]
    assert after == before


def test_database_execute_endpoint_rejects_raw_answer_table_writes(database_app) -> None:
    """Keep SQL console browsing available without bypassing workspace services."""

    response = _post_json(
        database_app,
        "/api/v1/database/execute",
        {"query": "UPDATE answer_workspace_state SET revision = revision + 1"},
    )

    assert response.status_code == 400
    assert "Answer workspace tables are managed" in response.json()["detail"]



def test_database_tables_exposes_only_application_tables(database_app) -> None:
    """Return schemas and counts without exposing SQLite's internal tables."""

    response = _request(database_app, "/api/v1/database/tables")

    assert response.status_code == 200
    tables = {table["name"]: table for table in response.json()["tables"]}
    assert set(tables) == {"query_history", "answer_workspace_state", "answer_candidates"}
    assert tables["query_history"]["row_count"] == 1
    assert tables["answer_candidates"]["row_count"] == 1
    assert "timestamp_ms" in {column["name"] for column in tables["answer_candidates"]["columns"]}
    state_columns = {column["name"] for column in tables["answer_workspace_state"]["columns"]}
    assert {"task_scope_key", "task_name"} <= state_columns
    assert "task_id" not in state_columns


def test_database_rows_returns_stable_bounded_raw_sqlite_page(database_app) -> None:
    """Return stored column values and pagination metadata unchanged."""

    response = _request(
        database_app,
        "/api/v1/database/tables/query_history/rows?page=1&page_size=1",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["table"] == "query_history"
    assert (payload["total_rows"], payload["total_pages"]) == (1, 1)
    assert payload["rows"][0]["query_id"] == "query-1"
    assert payload["rows"][0]["result_snapshot_json"] == '{"results":[]}'


def test_database_rows_rejects_unknown_tables_and_unbounded_pages(database_app) -> None:
    """Prevent arbitrary table access and oversized database responses."""

    unknown = _request(database_app, "/api/v1/database/tables/sqlite_master/rows")
    oversized = _request(
        database_app,
        "/api/v1/database/tables/query_history/rows?page_size=101",
    )

    assert unknown.status_code == 404
    assert oversized.status_code == 422


def test_database_router_reports_missing_sqlite_configuration() -> None:
    """Expose missing workspace storage as HTTP 503."""

    response = _request(create_app(), "/api/v1/database/tables")

    assert response.status_code == 503


def test_database_execute_endpoint_select(database_app) -> None:
    """Endpoint handles SELECT queries successfully."""

    response = _post_json(
        database_app,
        "/api/v1/database/execute",
        {"query": "SELECT query_id FROM query_history", "max_rows": 10},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"] == ["query_id"]
    assert len(payload["rows"]) == 1
    assert payload["is_mutation"] is False


def test_database_execute_endpoint_unrelated_mutation(database_app) -> None:
    """Keep database-console writes outside answer state available over HTTP."""

    response = _post_json(
        database_app,
        "/api/v1/database/execute",
        {"query": "UPDATE query_history SET query_text = 'changed' WHERE query_id = 'query-1'"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_mutation"] is True
    assert payload["rows_affected"] == 1


def test_database_execute_endpoint_syntax_error(database_app) -> None:
    """Endpoint returns HTTP 400 on SQLite syntax error."""

    response = _post_json(
        database_app,
        "/api/v1/database/execute",
        {"query": "SELCT * FROM query_history"},
    )
    assert response.status_code == 400
    assert "syntax error" in response.json()["detail"]

