"""Dependency-free process metrics for latency and stage failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock
from typing import Any

from hcmai.common.schemas import StageStatus, StageTrace

_BUCKETS_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._stage_counts: Counter[str] = Counter()
        self._failure_counts: Counter[str] = Counter()
        self._histograms: dict[str, Counter[str]] = defaultdict(Counter)

    def observe_stage(self, trace: StageTrace) -> None:
        with self._lock:
            self._stage_counts[trace.stage] += 1
            bucket = next(
                (str(value) for value in _BUCKETS_MS if trace.duration_ms <= value),
                "+Inf",
            )
            self._histograms[trace.stage][bucket] += 1
            if trace.status in {StageStatus.FAILED, StageStatus.PARTIAL}:
                category = trace.error_category or "unspecified"
                self._failure_counts[f"{trace.stage}:{category}"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stage_counts": dict(sorted(self._stage_counts.items())),
                "failure_counts": dict(sorted(self._failure_counts.items())),
                "latency_histograms_ms": {
                    stage: dict(sorted(buckets.items()))
                    for stage, buckets in sorted(self._histograms.items())
                },
            }


METRICS = MetricsRegistry()
