"""Shared executable interface for competition task pipelines."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import SearchRequest, SearchResponse, TaskType


class TaskPipelineDependencyError(RuntimeError):
    """A required dependency of an executable task pipeline is unavailable."""


class TaskPipeline(Protocol):
    """One executable pipeline registered for one task type."""

    @property
    def task_type(self) -> TaskType:
        """Return the task type handled by this pipeline."""

        ...

    def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute the pipeline for a validated public request."""

        ...
