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


@pytest.fixture
def database_app(tmp_path):
    """Create a populated temporary SQLite store behind the production router."""

    store = WorkspaceStore(tmp_path / "workspace.sqlite3")
    store.create_history(
        QueryHistoryCreate(
            query_id="query-1",
            user_id="user-1",
            query_text="red bus",
            result_snapshot={"results": []},
        )
    )
    store.create_submission_file("answer.csv", "L21_V001,90")
    return create_app(workspace_store=store)


def test_database_tables_exposes_only_application_tables(database_app) -> None:
    """Return schemas and counts without exposing SQLite's internal tables."""

    response = _request(database_app, "/api/v1/database/tables")

    assert response.status_code == 200
    tables = {table["name"]: table for table in response.json()["tables"]}
    assert set(tables) == {"query_history", "submission_files"}
    assert tables["query_history"]["row_count"] == 1
    assert {column["name"] for column in tables["submission_files"]["columns"]} == {
        "name",
        "content",
        "is_validated",
        "revision",
    }


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
