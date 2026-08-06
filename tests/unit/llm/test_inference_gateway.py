"""Deterministic resilience tests for the remote inference gateway."""

from __future__ import annotations

import httpx
import pytest

from hcmai.common.config import InferenceConfig
from hcmai.llm.adapters.http import InferenceClient, InferenceClientError
from hcmai.llm.gateway import InferenceGateway, InferenceGatewayError
from hcmai.llm.pipeline import LLMService
from hcmai.llm.resilience import CircuitState, FailureCategory
from hcmai.orchestration.pipeline import SearchService


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _config(**updates) -> InferenceConfig:
    values = {
        "enabled": True,
        "max_attempts": 3,
        "backoff_initial_seconds": 0.1,
        "backoff_max_seconds": 1.0,
        "backoff_jitter_ratio": 0.0,
        "circuit_failure_threshold": 2,
        "circuit_cooldown_seconds": 5.0,
        "pool_timeout_seconds": 0.01,
        "minimum_retry_budget_seconds": 0.05,
    }
    values.update(updates)
    return InferenceConfig(**values)


def _gateway(handler, config=None, fake_time=None):
    clock = fake_time or FakeTime()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://model.test",
    )
    return (
        InferenceGateway(
            "https://model.test",
            config or _config(),
            client,
            clock=clock,
            sleeper=clock.sleep,
        ),
        clock,
    )


def test_transient_idempotent_failures_retry_with_exponential_backoff() -> None:
    statuses = iter([503, 429, 200])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(next(statuses), json={"ready": True, "models": {}})

    gateway, clock = _gateway(handler)

    response = gateway.request("GET", "/ready", idempotent=True)

    assert response.status_code == 200
    assert len(calls) == 3
    assert clock.sleeps == pytest.approx([0.1, 0.2])
    assert gateway.circuit.state is CircuitState.CLOSED


def test_deterministic_client_error_and_non_idempotent_call_do_not_retry() -> None:
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"detail": "malformed"})

    gateway, clock = _gateway(handler)

    with pytest.raises(InferenceGatewayError) as caught:
        gateway.request("POST", "/v1/vqa", idempotent=True)

    assert caught.value.category is FailureCategory.CLIENT_ERROR
    assert caught.value.retryable is False
    assert calls == 1
    assert clock.sleeps == []

    def transient(_):
        return httpx.Response(503)

    gateway, _ = _gateway(transient)
    with pytest.raises(InferenceGatewayError) as non_idempotent:
        gateway.request("POST", "/mutating", idempotent=False)
    assert non_idempotent.value.attempt_count == 1


def test_timeout_is_bounded_and_opens_circuit_per_failed_logical_call() -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("late", request=request)

    gateway, _ = _gateway(handler, _config(max_attempts=2))

    for _ in range(2):
        with pytest.raises(InferenceGatewayError) as caught:
            gateway.request("GET", "/ready", idempotent=True)
        assert caught.value.category is FailureCategory.TIMEOUT
        assert caught.value.attempt_count == 2

    assert calls == 4
    assert gateway.circuit.state is CircuitState.OPEN
    with pytest.raises(InferenceGatewayError) as opened:
        gateway.request("GET", "/ready", idempotent=True)
    assert opened.value.category is FailureCategory.CIRCUIT_OPEN
    assert calls == 4


def test_half_open_probe_closes_circuit_after_cooldown() -> None:
    fail = True

    def handler(request):
        if fail:
            raise httpx.ReadError("reset", request=request)
        return httpx.Response(200, json={})

    clock = FakeTime()
    gateway, _ = _gateway(
        handler,
        _config(max_attempts=1, circuit_failure_threshold=1),
        clock,
    )
    with pytest.raises(InferenceGatewayError):
        gateway.request("GET", "/ready", idempotent=True)
    assert gateway.circuit.state is CircuitState.OPEN

    clock.now += 5.0
    assert gateway.circuit.state is CircuitState.HALF_OPEN
    fail = False
    assert gateway.request("GET", "/ready", idempotent=True).status_code == 200
    assert gateway.circuit.state is CircuitState.CLOSED
    assert gateway.circuit.failure_count == 0


def test_deadline_prevents_retry_without_enough_backoff_budget() -> None:
    calls = 0

    def handler(_):
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    clock = FakeTime()
    gateway, _ = _gateway(handler, fake_time=clock)

    with pytest.raises(InferenceGatewayError) as caught:
        gateway.request(
            "GET",
            "/ready",
            idempotent=True,
            deadline_at=0.12,
        )

    assert caught.value.category is FailureCategory.SERVER_ERROR
    assert caught.value.attempt_count == 1
    assert calls == 1
    assert clock.sleeps == []


def test_bulkhead_rejects_when_shared_capacity_is_exhausted() -> None:
    gateway, _ = _gateway(lambda _: httpx.Response(200), _config(max_concurrency=1))
    assert gateway.bulkhead.acquire(0)
    try:
        with pytest.raises(InferenceGatewayError) as caught:
            gateway.request("GET", "/ready", idempotent=True)
    finally:
        gateway.bulkhead.release()

    assert caught.value.category is FailureCategory.BULKHEAD_FULL


def test_separate_http_timeouts_and_safe_client_error_metadata() -> None:
    observed = {}

    def timeout_handler(request):
        observed.update(request.extensions["timeout"])
        return httpx.Response(400, json={"detail": "secret backend response"})

    config = _config(
        max_attempts=1,
        connect_timeout_seconds=1,
        read_timeout_seconds=2,
        write_timeout_seconds=3,
        pool_timeout_seconds=4,
    )
    gateway, _ = _gateway(timeout_handler, config)
    client = InferenceClient("https://model.test", config, gateway=gateway)

    with pytest.raises(InferenceClientError) as caught:
        client.embed_text(["query"])

    assert observed == {"connect": 1, "read": 2, "write": 3, "pool": 4}
    assert caught.value.category is FailureCategory.CLIENT_ERROR
    assert caught.value.attempt_count == 1
    assert "secret backend response" not in str(caught.value)


def test_capability_discovery_contract_is_preserved() -> None:
    def handler(_):
        return httpx.Response(200, json={
            "ready": True,
            "models": {},
            "capabilities": {
                "embedding": True,
                "reranking": True,
                "multi_image_vqa": False,
                "structured_parsing": True,
            },
        })

    gateway, _ = _gateway(handler)
    client = InferenceClient("https://model.test", gateway=gateway)

    readiness = client.readiness()

    assert readiness.capabilities.embedding is True
    assert readiness.capabilities.reranking is True
    assert readiness.capabilities.multi_image_vqa is False
    assert client.health()["circuit_state"] == "closed"

    system_health = SearchService(None, None, llm=LLMService(client)).health()
    assert system_health["remote_inference"]["circuit_state"] == "closed"
