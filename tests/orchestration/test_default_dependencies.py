"""Tests for default temporal-baseline dependency wiring.

These tests exercise startup composition with loader fakes so the default KIS
and TRAKE runtime contract can be checked without loading production artifacts.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

from hcmai.common.config import AppConfig
from hcmai.common.schemas import RetrievalSource


if "faiss" not in sys.modules:
    faiss_stub = ModuleType("faiss")
    faiss_stub.IndexFlatIP = object
    faiss_stub.METRIC_INNER_PRODUCT = 0
    faiss_stub.read_index = lambda *args, **kwargs: None
    faiss_stub.write_index = lambda *args, **kwargs: None
    sys.modules["faiss"] = faiss_stub

from hcmai.orchestration import setup


def test_load_search_service_has_visual_only_requirements_and_no_reranker(
    monkeypatch,
) -> None:
    """Build the default service without attaching any reranking dependency."""

    settings = AppConfig()
    fake_data = cast(Any, SimpleNamespace())
    fake_retrieval = cast(
        Any,
        SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(setup, "_load_app_config", lambda: settings)
    monkeypatch.setattr(setup, "_load_model_config", lambda: SimpleNamespace())
    monkeypatch.setattr(setup, "_load_data", lambda *args, **kwargs: fake_data)
    monkeypatch.setattr(setup, "_load_remote_llm", lambda *args, **kwargs: None)

    def load_retrieval(
        settings_arg: AppConfig,
        models_arg: object,
        index_dir: object,
        llm: object,
        messages: list[str],
        *,
        data: object,
    ) -> object:
        """Capture the default required sources while returning a fake retriever."""

        del models_arg, index_dir, llm, messages
        captured["required_sources"] = settings_arg.search.fusion.required_sources
        captured["data"] = data
        return fake_retrieval

    monkeypatch.setattr(setup, "_load_retrieval", load_retrieval)

    service = setup.load_search_service(messages=[])

    assert captured["required_sources"] == {RetrievalSource.VISUAL}
    assert captured["data"] is fake_data
    assert service.retrieval is fake_retrieval
    assert not hasattr(service, "reranking")
    assert not hasattr(service.kis, "reranking")
    assert not hasattr(service.trake, "reranking")
    assert service.kis.temporal is service.trake.temporal
