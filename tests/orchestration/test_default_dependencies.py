"""Tests for default temporal-baseline dependency wiring.

These tests exercise startup composition with loader fakes so the default KIS
and TRAKE runtime contract can be checked without loading production artifacts.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from hcmai.common.config import AppConfig
from hcmai.retrieval.models import RetrievalSource


_STUBBED_IMPORT_MODULES = (
    "hcmai.orchestration.setup",
    "hcmai.retrieval.retriever.pipeline",
    "hcmai.retrieval.retriever.dense.index",
    "hcmai.retrieval.retriever.segment.index",
)


def _unload_setup_modules() -> None:
    """Remove modules imported under a temporary FAISS stub."""

    for module_name in _STUBBED_IMPORT_MODULES:
        sys.modules.pop(module_name, None)


def _import_setup_module() -> Any:
    """Import ``hcmai.orchestration.setup`` with scoped FAISS fallback only.

    When FAISS is installed, the real package is imported directly. When it is
    unavailable in this test environment, a minimal stub exists only during the
    import path that needs it and is removed immediately afterwards.
    """

    if importlib.util.find_spec("faiss") is not None:
        return importlib.import_module("hcmai.orchestration.setup")

    faiss_stub = ModuleType("faiss")
    faiss_stub.IndexFlatIP = object
    faiss_stub.METRIC_INNER_PRODUCT = 0
    faiss_stub.read_index = lambda *args, **kwargs: None
    faiss_stub.write_index = lambda *args, **kwargs: None

    _unload_setup_modules()
    with patch.dict(sys.modules, {"faiss": faiss_stub}):
        return importlib.import_module("hcmai.orchestration.setup")


def test_load_search_service_has_visual_only_requirements_and_no_reranker(
    monkeypatch,
) -> None:
    """Build the default service without attaching any reranking dependency."""

    settings = AppConfig()
    fake_corpus = cast(Any, SimpleNamespace())
    fake_retrieval = cast(
        Any,
        SimpleNamespace(active_sources=(RetrievalSource.VISUAL,)),
    )
    captured: dict[str, Any] = {}
    try:
        setup = _import_setup_module()

        monkeypatch.setattr(setup, "_load_app_config", lambda: settings)
        monkeypatch.setattr(setup, "_load_model_config", lambda: SimpleNamespace())
        monkeypatch.setattr(
            setup, "_load_corpus", lambda *args, **kwargs: fake_corpus
        )
        monkeypatch.setattr(setup, "_load_remote_llm", lambda *args, **kwargs: None)

        def load_retrieval(
            settings_arg: AppConfig,
            models_arg: object,
            index_dir: object,
            llm: object,
            messages: list[str],
            *,
            corpus: object,
        ) -> object:
            """Capture required sources while returning a fake retriever."""

            del models_arg, index_dir, llm, messages
            captured["required_sources"] = settings_arg.search.fusion.required_sources
            captured["corpus"] = corpus
            return fake_retrieval

        monkeypatch.setattr(setup, "_load_retrieval", load_retrieval)

        service = setup.load_search_service(messages=[])

        assert captured["required_sources"] == {RetrievalSource.VISUAL}
        assert captured["corpus"] is fake_corpus
        assert service.retrieval is fake_retrieval
        assert not hasattr(service, "reranking")
        assert not hasattr(service.kis, "reranking")
        assert not hasattr(service.trake, "reranking")
        assert service.kis.temporal is service.trake.temporal
    finally:
        _unload_setup_modules()
