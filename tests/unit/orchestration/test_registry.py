"""Composition tests for the shared stateless temporal alignment registry."""

from __future__ import annotations

from typing import cast

from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.common.schemas import TaskType


class _Data:
    """Placeholder canonical-data dependency used only for registry assembly."""


class _Retrieval:
    """Placeholder retrieval dependency used only for registry assembly."""


def test_default_registry_shares_alignment_without_default_reranking() -> None:
    """Keep task-head composition independent of the optional reranking package."""

    service = SearchService(
        cast(DataService, _Data()),
        cast(RetrievalService, _Retrieval()),
    )
    kis = service.pipeline_registry.get(TaskType.KIS)
    trake = service.pipeline_registry.get(TaskType.TRAKE)

    assert kis.alignment is trake.alignment
    assert not hasattr(kis, "reranking")
    assert not hasattr(service, "reranking")
