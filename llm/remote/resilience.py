"""Thread-safe retry, circuit-breaker, and bulkhead primitives for remote inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from threading import BoundedSemaphore, Lock
from typing import Callable

from hcmai.common.config import InferenceConfig


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureCategory(str, Enum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    INVALID_RESPONSE = "invalid_response"
    CIRCUIT_OPEN = "circuit_open"
    BULKHEAD_FULL = "bulkhead_full"
    DEADLINE_EXCEEDED = "deadline_exceeded"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    initial_seconds: float
    maximum_seconds: float
    jitter_ratio: float
    minimum_budget_seconds: float

    @classmethod
    def from_config(cls, config: InferenceConfig) -> RetryPolicy:
        return cls(
            max_attempts=config.max_attempts,
            initial_seconds=config.backoff_initial_seconds,
            maximum_seconds=config.backoff_max_seconds,
            jitter_ratio=config.backoff_jitter_ratio,
            minimum_budget_seconds=config.minimum_retry_budget_seconds,
        )

    def delay(self, completed_attempts: int, random: Random) -> float:
        base = min(
            self.maximum_seconds,
            self.initial_seconds * (2 ** max(0, completed_attempts - 1)),
        )
        if base == 0 or self.jitter_ratio == 0:
            return base
        return base * random.uniform(
            1 - self.jitter_ratio,
            1 + self.jitter_ratio,
        )


class CircuitBreaker:
    """Allow one half-open probe after a bounded cooldown."""

    def __init__(
        self,
        failure_threshold: int,
        cooldown_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state is CircuitState.OPEN
                and self._opened_at is not None
                and self._clock() - self._opened_at >= self.cooldown_seconds
            ):
                return CircuitState.HALF_OPEN
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failures

    def acquire_permission(self) -> bool:
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.OPEN:
                assert self._opened_at is not None
                if self._clock() - self._opened_at < self.cooldown_seconds:
                    return False
                self._state = CircuitState.HALF_OPEN
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._probe_in_flight = False
            if (
                self._state is CircuitState.HALF_OPEN
                or self._failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()


class Bulkhead:
    """Bound the number of logical remote requests in flight."""

    def __init__(self, maximum: int) -> None:
        self._semaphore = BoundedSemaphore(maximum)

    def acquire(self, timeout_seconds: float) -> bool:
        return self._semaphore.acquire(timeout=max(0.0, timeout_seconds))

    def release(self) -> None:
        self._semaphore.release()
