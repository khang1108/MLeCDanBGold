from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def inline_router_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ASGI contract tests deterministic on Python 3.14.

    Python 3.14 currently deadlocks while shutting down the AnyIO worker used by
    Starlette's test transport. Production routes still use the real bounded
    threadpool; HTTP tests replace only the scheduling boundary while retaining
    request validation, routing, error mapping, and response serialization.
    """

    async def run_inline(
        function: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr("hcmai.api.routers.search.run_in_threadpool", run_inline)
    monkeypatch.setattr("hcmai.api.routers.trake.run_in_threadpool", run_inline)
    monkeypatch.setattr("hcmai.api.routers.database.run_in_threadpool", run_inline)
