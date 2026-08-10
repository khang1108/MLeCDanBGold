"""Shared executable interface for competition task pipelines."""

from __future__ import annotations

from typing import Protocol

from hcmai.common.schemas import TaskRequest, TaskResponse, TaskType


class TaskPipelineDependencyError(RuntimeError):
    """A required dependency of an executable task pipeline is unavailable."""


class TaskPipelineRequestError(ValueError):
    """A validated task request is incompatible with the selected pipeline."""


class TaskPipeline(Protocol):
    """One executable pipeline registered for one task type."""

    @property
    def task_type(self) -> TaskType:
        """Return the task type handled by this pipeline."""

        ...

    def execute(self, request: TaskRequest) -> TaskResponse:
        """Execute the pipeline for a validated public task request."""

        ...
