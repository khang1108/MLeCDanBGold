"""Helpers that create and log request-scoped stage traces."""

from __future__ import annotations

import json
import logging
from time import perf_counter

from hcmai.common.observability.metrics import METRICS
from hcmai.common.schemas import StageStatus, StageTrace, TaskType


class StageTimer:
    """Measure one stage without storing timing on a shared service object."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.started_at = perf_counter()

    def finish(
        self,
        *,
        status: StageStatus = StageStatus.SUCCESS,
        attempt_count: int = 1,
        cache_hit: bool = False,
        error_category: str | None = None,
        input_count: int | None = None,
        output_count: int | None = None,
        backend: str | None = None,
        fallback_used: bool = False,
    ) -> StageTrace:
        """Finish this timer and return a request-owned trace value."""

        ended_at = perf_counter()
        return StageTrace(
            stage=self.stage,
            started_at=self.started_at,
            ended_at=ended_at,
            duration_ms=max(0.0, (ended_at - self.started_at) * 1_000),
            status=status,
            attempt_count=attempt_count,
            cache_hit=cache_hit,
            error_category=error_category,
            input_count=input_count,
            output_count=output_count,
            backend=backend,
            fallback_used=fallback_used,
        )


def log_stage(
    logger: logging.Logger,
    *,
    request_id: str,
    task_type: TaskType | str,
    trace: StageTrace,
) -> None:
    """Emit the required stage fields as one deterministic JSON log record."""

    task_value = getattr(task_type, "value", task_type)
    METRICS.observe_stage(trace)
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "task_type": task_value,
                "stage": trace.stage,
                "duration_ms": trace.duration_ms,
                "status": trace.status.value,
                "input_count": trace.input_count,
                "output_count": trace.output_count,
                "backend": trace.backend,
                "fallback_used": trace.fallback_used,
                "error_category": trace.error_category,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
