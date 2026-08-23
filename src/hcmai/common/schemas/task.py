"""Discriminated unions for task-specific public contracts."""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias

from pydantic import Discriminator, Tag

from .enum import TaskType
from .search import SearchRequest, SearchResponse
from .trake import TRAKERequest, TRAKEResponse


def _task_discriminator(value: Any) -> str | None:
    """Map request/response values to one union branch."""

    if isinstance(value, dict):
        query_type = value.get("query_type", TaskType.KIS)
    else:
        query_type = getattr(value, "query_type", TaskType.KIS)
    raw_value = getattr(query_type, "value", query_type)
    if raw_value == TaskType.KIS.value:
        return "search"
    if raw_value == TaskType.TRAKE.value:
        return "trake"
    return None


TaskRequest: TypeAlias = Annotated[
    Annotated[SearchRequest, Tag("search")]
    | Annotated[TRAKERequest, Tag("trake")],
    Discriminator(_task_discriminator),
]

TaskResponse: TypeAlias = Annotated[
    Annotated[SearchResponse, Tag("search")]
    | Annotated[TRAKEResponse, Tag("trake")],
    Discriminator(_task_discriminator),
]
