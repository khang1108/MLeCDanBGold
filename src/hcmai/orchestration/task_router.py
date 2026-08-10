"""Task-pipeline registration and deterministic lookup."""

from __future__ import annotations

from collections.abc import Iterable

from hcmai.common.schemas import TaskType
from hcmai.orchestration.workflows.base import TaskPipeline


class PipelineRegistry:
    """Own the unique executable pipeline registered for each task type."""

    def __init__(self, pipelines: Iterable[TaskPipeline] = ()) -> None:
        self._pipelines: dict[TaskType, TaskPipeline] = {}
        for pipeline in pipelines:
            self.register(pipeline)

    def register(self, pipeline: TaskPipeline) -> None:
        """Register a pipeline, rejecting ambiguous duplicate ownership."""

        task_type = pipeline.task_type
        if task_type in self._pipelines:
            raise ValueError(
                f"pipeline for task_type {task_type.value!r} is already registered"
            )
        self._pipelines[task_type] = pipeline

    def get(self, task_type: TaskType) -> TaskPipeline:
        """Return the pipeline for ``task_type`` or raise ``KeyError``."""

        return self._pipelines[task_type]

    def capability_report(
        self, task_types: Iterable[TaskType] | None = None
    ) -> dict[str, bool]:
        """Report registration status in stable caller-supplied order."""

        requested = TaskType if task_types is None else task_types
        return {
            task_type.value: task_type in self._pipelines
            for task_type in requested
        }
