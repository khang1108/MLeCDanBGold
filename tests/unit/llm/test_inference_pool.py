from __future__ import annotations

from types import SimpleNamespace

import pytest

from hcmai.common.schemas import InferenceReadiness
from thundercompute.adapters.pool import InferenceClientPool


class FakeClient:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls = 0

    def readiness(self):
        if self.error:
            raise self.error
        return InferenceReadiness(ready=True, models={})

    def caption(self, images):
        self.calls += 1
        if self.error:
            raise self.error
        return self.value

    def close(self):
        return None


def test_pool_fails_over_to_another_semantically_compatible_client() -> None:
    failed = FakeClient(error=RuntimeError("session expired"))
    healthy = FakeClient(value=SimpleNamespace(items=[]))
    pool = InferenceClientPool([failed, healthy])

    assert pool.caption([]) is healthy.value
    assert failed.calls == 1
    assert healthy.calls == 1


def test_pool_reports_no_ready_endpoint_without_leaking_provider_detail() -> None:
    pool = InferenceClientPool([
        FakeClient(error=RuntimeError("first")),
        FakeClient(error=RuntimeError("second")),
    ])
    with pytest.raises(RuntimeError, match="no inference endpoint is ready"):
        pool.readiness()

