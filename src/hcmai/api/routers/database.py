"""Read-only HTTP router for the configured workspace SQLite database.

Only application-owned tables are exposed. The router accepts pagination but
does not accept SQL, database paths, column names, or sort expressions.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from hcmai.api.contracts import (
    DatabaseQueryRequest,
    DatabaseQueryResponse,
    DatabaseRowsPage,
    DatabaseTableList,
)
from hcmai.api.history import WorkspaceStore


def _workspace_store(container: dict[str, Any]) -> WorkspaceStore:
    """Return the configured SQLite store or an HTTP 503 response."""

    store = container.get("workspace_store")
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workspace database is not configured",
        )
    return store


def create_database_router(service_container: dict[str, Any]) -> APIRouter:
    """Create table-inventory and paginated-row endpoints for SQLite."""

    router = APIRouter(prefix="/api/v1/database", tags=["database"])

    @router.get("/tables", response_model=DatabaseTableList)
    async def list_tables() -> DatabaseTableList:
        """Return allowlisted tables, columns, and current row counts."""

        tables = await run_in_threadpool(
            _workspace_store(service_container).list_database_tables
        )
        return DatabaseTableList(tables=tables)

    @router.get("/tables/{table_name}/rows", response_model=DatabaseRowsPage)
    async def list_rows(
        table_name: str,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> DatabaseRowsPage:
        """Return one bounded page from an allowlisted SQLite table."""

        try:
            return await run_in_threadpool(
                _workspace_store(service_container).list_database_rows,
                table_name,
                page=page,
                page_size=page_size,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.post("/execute", response_model=DatabaseQueryResponse)
    async def execute_query(request: DatabaseQueryRequest) -> DatabaseQueryResponse:
        """Execute arbitrary SQL query against the workspace SQLite database."""

        try:
            return await run_in_threadpool(
                _workspace_store(service_container).execute_query,
                request.query,
                max_rows=request.max_rows,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    return router
