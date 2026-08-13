"""Workflow interfaces and compatibility exports used by orchestration."""

from typing import TYPE_CHECKING

from hcmai.orchestration.workflows.base import (
    TaskPipeline,
    TaskPipelineDependencyError,
    TaskPipelineRequestError,
)
from hcmai.orchestration.workflows.kis import KISPipeline
from hcmai.orchestration.workflows.trake import TRAKEPipeline

if TYPE_CHECKING:
    from hcmai.pipelines.vqa.pipeline import VQAPipeline

__all__ = [
    "KISPipeline",
    "TRAKEPipeline",
    "VQAPipeline",
    "TaskPipeline",
    "TaskPipelineDependencyError",
    "TaskPipelineRequestError",
]


def __getattr__(name: str):
    """Lazily preserve the former VQAPipeline export without a cycle."""

    if name == "VQAPipeline":
        from hcmai.pipelines.vqa.pipeline import VQAPipeline

        return VQAPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
