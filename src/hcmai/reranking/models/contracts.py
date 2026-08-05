"""Model-provider boundary consumed by :mod:`hcmai.reranking.pipeline`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RerankingAdapter(Protocol):
    """Score an ordered batch without changing item identity or order."""

    def score(self, query: str, images: Sequence[Any]) -> Sequence[float]: ...


class HostedRerankingAdapter(RerankingAdapter, Protocol):
    """Local model adapter lifecycle used by the private inference server."""

    resolved_revision: str | None
    _base_model: Any | None

    def _ensure_loaded(self) -> None: ...

    def score_batch(
        self, query: str, images: Sequence[Any]
    ) -> Sequence[float]: ...
