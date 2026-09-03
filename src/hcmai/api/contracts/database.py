"""HTTP contracts for safe, read-only inspection of the workspace SQLite DB.

The contracts expose table metadata and paginated rows. They do not expose the
database path, SQLite connection details, or arbitrary query execution.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class DatabaseColumn(BaseModel):
    """One SQLite column projected for frontend table headers."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: str
    nullable: bool
    primary_key: bool


class DatabaseTable(BaseModel):
    """Metadata and current row count for one frontend-visible table."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    columns: list[DatabaseColumn]


class DatabaseTableList(BaseModel):
    """Allowlisted SQLite tables available to the frontend browser."""

    model_config = ConfigDict(extra="forbid")

    tables: list[DatabaseTable]


class DatabaseRowsPage(BaseModel):
    """One bounded page of raw SQLite values from an allowlisted table."""

    model_config = ConfigDict(extra="forbid")

    table: str = Field(min_length=1)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_rows: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    rows: list[dict[str, JsonValue]] = Field(default_factory=list)


__all__ = [
    "DatabaseColumn",
    "DatabaseRowsPage",
    "DatabaseTable",
    "DatabaseTableList",
]
