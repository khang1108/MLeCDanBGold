"""Resilient gateway for bounded remote inference requests."""

from __future__ import annotations

from random import Random
from time import monotonic, sleep
from typing import Any, Callable

import httpx

from hcmai.common.config import InferenceConfig
from hcmai.common.utils.logging import get_logger
from thundercompute.resilience import (
    Bulkhead,
    CircuitBreaker,
    FailureCategory,
    RetryPolicy,
)

logger = get_logger(__name__)

_TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


class InferenceGatewayError(RuntimeError):
    """Safe categorized failure from one logical remote request."""

    def __init__(
        self,
        category: FailureCategory,
        *,
        attempt_count: int,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"remote inference failed ({category.value})")
        self.category = category
        self.attempt_count = attempt_count
        self.retryable = retryable
        self.status_code = status_code


class InferenceGateway:
    """Apply timeouts, retry, circuit breaking, and a shared bulkhead."""

    def __init__(
        self,
        base_url: str,
        config: InferenceConfig,
        client: httpx.Client | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
        random: Random | None = None,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleep = sleeper
        self._random = random or Random()
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=self._timeout(None),
        )
        self.retry_policy = RetryPolicy.from_config(config)
        self.circuit = CircuitBreaker(
            config.circuit_failure_threshold,
            config.circuit_cooldown_seconds,
            clock,
        )
        self.bulkhead = Bulkhead(config.max_concurrency)

    def request(
        self,
        method: str,
        path: str,
        *,
        idempotent: bool,
        deadline_at: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute one logical call without exceeding its retry/deadline budget."""

        acquire_timeout = self.config.pool_timeout_seconds
        remaining = self._remaining(deadline_at)
        if remaining is not None:
            if remaining <= 0:
                raise InferenceGatewayError(
                    FailureCategory.DEADLINE_EXCEEDED,
                    attempt_count=0,
                    retryable=False,
                )
            acquire_timeout = min(acquire_timeout, remaining)
        if not self.bulkhead.acquire(acquire_timeout):
            raise InferenceGatewayError(
                FailureCategory.BULKHEAD_FULL,
                attempt_count=0,
                retryable=False,
            )
        try:
            if not self.circuit.acquire_permission():
                raise InferenceGatewayError(
                    FailureCategory.CIRCUIT_OPEN,
                    attempt_count=0,
                    retryable=False,
                )
            return self._request_with_retries(
                method,
                path,
                idempotent=idempotent,
                deadline_at=deadline_at,
                **kwargs,
            )
        finally:
            self.bulkhead.release()

    def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        idempotent: bool,
        deadline_at: float | None,
        **kwargs: Any,
    ) -> httpx.Response:
        attempts = 0
        last_error: InferenceGatewayError | None = None
        while attempts < self.retry_policy.max_attempts:
            remaining = self._remaining(deadline_at)
            if remaining is not None and remaining <= 0:
                last_error = InferenceGatewayError(
                    FailureCategory.DEADLINE_EXCEEDED,
                    attempt_count=attempts,
                    retryable=False,
                )
                break
            attempts += 1
            try:
                response = self.client.request(
                    method,
                    path,
                    timeout=self._timeout(remaining),
                    **kwargs,
                )
                error = _status_error(response, attempts)
                if error is not None:
                    raise error
                self.circuit.record_success()
                return response
            except InferenceGatewayError as error:
                last_error = error
            except httpx.TimeoutException:
                last_error = InferenceGatewayError(
                    FailureCategory.TIMEOUT,
                    attempt_count=attempts,
                    retryable=True,
                )
            except httpx.NetworkError:
                last_error = InferenceGatewayError(
                    FailureCategory.CONNECTION,
                    attempt_count=attempts,
                    retryable=True,
                )

            assert last_error is not None
            if (
                not idempotent
                or not last_error.retryable
                or attempts >= self.retry_policy.max_attempts
            ):
                break
            delay = self.retry_policy.delay(attempts, self._random)
            remaining = self._remaining(deadline_at)
            if remaining is not None and (
                remaining < delay + self.retry_policy.minimum_budget_seconds
            ):
                break
            self._sleep(delay)

        assert last_error is not None
        if last_error.retryable:
            self.circuit.record_failure()
        else:
            self.circuit.record_success()
        raise last_error

    def health(self) -> dict[str, Any]:
        return {
            "configured": True,
            "circuit_state": self.circuit.state.value,
            "consecutive_failures": self.circuit.failure_count,
            "max_attempts": self.retry_policy.max_attempts,
            "max_concurrency": self.config.max_concurrency,
        }

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _remaining(self, deadline_at: float | None) -> float | None:
        if deadline_at is None:
            return None
        return deadline_at - self._clock()

    def _timeout(self, remaining: float | None) -> httpx.Timeout:
        def bounded(value: float) -> float:
            return value if remaining is None else max(0.001, min(value, remaining))

        return httpx.Timeout(
            connect=bounded(self.config.connect_timeout_seconds),
            read=bounded(self.config.read_timeout_seconds),
            write=bounded(self.config.write_timeout_seconds),
            pool=bounded(self.config.pool_timeout_seconds),
        )


def _status_error(
    response: httpx.Response,
    attempt_count: int,
) -> InferenceGatewayError | None:
    status = response.status_code
    if status < 400:
        return None
    if status == 429:
        category = FailureCategory.RATE_LIMITED
    elif status in {502, 503, 504}:
        category = FailureCategory.SERVER_ERROR
    elif 400 <= status < 500:
        category = FailureCategory.CLIENT_ERROR
    else:
        category = FailureCategory.SERVER_ERROR
    # Log the response body so the actual server-side error message is visible,
    # not just the HTTP status category.
    try:
        detail = response.json().get("detail", response.text[:300])
    except Exception:
        detail = response.text[:300]
    logger.warning(
        "Remote inference error status=%d category=%s detail=%r",
        status, category.value, detail,
    )
    return InferenceGatewayError(
        category,
        attempt_count=attempt_count,
        retryable=status in _TRANSIENT_STATUS_CODES,
        status_code=status,
    )
