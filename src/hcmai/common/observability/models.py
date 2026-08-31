"""Request-scoped observability values.

The models are immutable runtime dataclasses. They validate stage outcomes but
do not own persistence, metrics emission, or log serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Self


_LEGACY_STAGE_ALIASES = {
    "query_encoding": "encode",
    "index_search": "search",
    "reranking": "rerank",
}


class StageStatus(str, Enum):
    """Outcome of one bounded pipeline stage."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class StageTrace:
    """Timing and outcome recorded for one request-local stage execution."""

    stage: str
    started_at: float
    ended_at: float
    duration_ms: float
    status: StageStatus
    attempt_count: int = 1
    cache_hit: bool = False
    error_category: str | None = None
    input_count: int | None = None
    output_count: int | None = None
    backend: str | None = None
    fallback_used: bool = False

    def __post_init__(self) -> None:
        """Validate timing, counts, and status-specific diagnostics."""

        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if min(self.started_at, self.ended_at, self.duration_ms) < 0:
            raise ValueError("trace timing values must be non-negative")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.attempt_count < 0:
            raise ValueError("attempt_count must be non-negative")
        if self.input_count is not None and self.input_count < 0:
            raise ValueError("input_count must be non-negative")
        if self.output_count is not None and self.output_count < 0:
            raise ValueError("output_count must be non-negative")
        if self.status is StageStatus.FAILED and self.error_category is None:
            raise ValueError("failed stages require an error_category")
        if (
            self.status in {StageStatus.SUCCESS, StageStatus.SKIPPED}
            and self.error_category is not None
        ):
            raise ValueError(
                "successful or skipped stages cannot define an error_category"
            )


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """Named request-local stages with deterministic aggregate helpers."""

    stages: dict[str, StageTrace] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require every mapping key to match its stored stage name."""

        mismatches = [
            name for name, trace in self.stages.items() if name != trace.stage
        ]
        if mismatches:
            names = ", ".join(sorted(mismatches))
            raise ValueError(f"trace keys must match stage names: {names}")

    @property
    def total_duration_ms(self) -> float:
        """Return the sum of recorded stage durations."""

        return sum(stage.duration_ms for stage in self.stages.values())

    def duration_for(self, stage_name: str) -> float:
        """Sum exact and modality-prefixed occurrences of ``stage_name``."""

        canonical_name = _LEGACY_STAGE_ALIASES.get(stage_name, stage_name)
        names = {
            stage_name,
            canonical_name,
            *(
                legacy_name
                for legacy_name, canonical in _LEGACY_STAGE_ALIASES.items()
                if canonical == canonical_name
            ),
        }
        return sum(
            trace.duration_ms
            for name, trace in self.stages.items()
            if any(name == value or name.endswith(f".{value}") for value in names)
        )

    def merged(
        self,
        other: PipelineTrace,
        *,
        prefix: str | None = None,
    ) -> Self:
        """Return a new trace containing every uniquely named stage."""

        additions = {
            f"{prefix}.{name}" if prefix else name: replace(
                trace,
                stage=f"{prefix}.{name}" if prefix else name,
            )
            for name, trace in other.stages.items()
        }
        duplicates = set(self.stages).intersection(additions)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate pipeline trace stages: {names}")
        return type(self)(stages={**self.stages, **additions})


@dataclass(frozen=True, slots=True)
class RetrievalTrace(PipelineTrace):
    """Pipeline trace produced by one retrieval call."""


__all__ = [
    "PipelineTrace",
    "RetrievalTrace",
    "StageStatus",
    "StageTrace",
]
