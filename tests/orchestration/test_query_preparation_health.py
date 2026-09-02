"""Tests for independent query-preparation setup and health."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from hcmai.common.config import AppConfig
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.setup import _load_query_preparation
from hcmai.query_preparation.service import QueryPreparationService


class FakeCorpus:
    """Minimal corpus surface required by SearchService health."""

    @staticmethod
    def __len__() -> int:
        return 1

    @staticmethod
    def has_evidence(source: Any) -> bool:
        return False

    @staticmethod
    def frame_asset_status() -> Any:
        return SimpleNamespace(
            as_dict=lambda: {"ready": True, "checked": 1, "available": 1, "missing": 0}
        )



class FakeLLM:
    """Advertise a configurable query-preparation capability."""

    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def capability_health(self) -> dict[str, bool]:
        return {"query_preparation": self.ready}


def test_query_preparation_is_independent_from_dense_search() -> None:
    """Keep search ready when the optional Qwen capability is absent."""

    service = SearchService(
        corpus=cast(Any, FakeCorpus()),
        retrieval=cast(Any, SimpleNamespace(active_sources=())),
    )

    health = service.health()

    assert health["capabilities"]["search"] is True
    assert health["capabilities"]["query_preparation"] is False


def test_setup_constructs_service_only_for_ready_capability() -> None:
    """Build query preparation only after remote readiness confirms support."""

    settings = AppConfig()
    messages: list[str] = []

    unavailable = _load_query_preparation(settings, cast(Any, FakeLLM(False)), messages)
    available = _load_query_preparation(settings, cast(Any, FakeLLM(True)), messages)

    assert unavailable is None
    assert isinstance(available, QueryPreparationService)
    assert any("Query preparation unavailable" in message for message in messages)