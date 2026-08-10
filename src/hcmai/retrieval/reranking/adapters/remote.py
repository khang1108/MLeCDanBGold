"""Remote inference adapter for reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class RemoteRerankingClient(Protocol):
    def rerank(self, query: str, images: Sequence[Any]) -> list[float]: ...


class RemoteAdapter:
    """Delegate ordered scoring to the configured inference client."""

    def __init__(self, client: RemoteRerankingClient) -> None:
        self.client = client

    def score(self, query: str, images: Sequence[Any]) -> list[float]:
        return self.client.rerank(query, images)
