"""Request-scoped pipeline telemetry contracts."""

from __future__ import annotations

from enum import Enum
from typing import Self

from pydantic import Field, model_validator

from .base import ContractModel, NonEmptyString


class StageStatus(str, Enum):
    """Outcome of one bounded pipeline stage."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


class StageTrace(ContractModel):
    """Timing and outcome recorded for one request-local stage execution."""

    stage: NonEmptyString
    started_at: float = Field(ge=0)
    ended_at: float = Field(ge=0)
    duration_ms: float = Field(ge=0)
    status: StageStatus
    attempt_count: int = Field(default=1, ge=0)
    cache_hit: bool = False
    error_category: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.status is StageStatus.FAILED and self.error_category is None:
            raise ValueError("failed stages require an error_category")
        if (
            self.status in {StageStatus.SUCCESS, StageStatus.SKIPPED}
            and self.error_category is not None
        ):
            raise ValueError(
                "successful or skipped stages cannot define an error_category"
            )
        return self


class PipelineTrace(ContractModel):
    """Named request-local stages with deterministic aggregate helpers."""

    stages: dict[str, StageTrace] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stage_names(self) -> Self:
        mismatches = [
            name for name, trace in self.stages.items() if name != trace.stage
        ]
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise ValueError(f"trace keys must match stage names: {names}")
        return self

    @property
    def total_duration_ms(self) -> float:
        """Return the sum of recorded stage durations."""

        return sum(stage.duration_ms for stage in self.stages.values())

    def duration_for(self, stage_name: str) -> float:
        """Sum exact and modality-prefixed occurrences of ``stage_name``."""

        suffix = f".{stage_name}"
        return sum(
            trace.duration_ms
            for name, trace in self.stages.items()
            if name == stage_name or name.endswith(suffix)
        )

    def merged(
        self,
        other: PipelineTrace,
        *,
        prefix: str | None = None,
    ) -> Self:
        """Return a new trace containing every uniquely named stage."""

        additions = {
            f"{prefix}.{name}" if prefix else name: trace.model_copy(
                update={"stage": f"{prefix}.{name}" if prefix else name}
            )
            for name, trace in other.stages.items()
        }
        duplicates = set(self.stages).intersection(additions)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate pipeline trace stages: {names}")
        return self.model_copy(update={"stages": {**self.stages, **additions}})


class RetrievalTrace(PipelineTrace):
    """Pipeline trace produced by one retrieval call."""
