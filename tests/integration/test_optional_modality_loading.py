"""Startup composition keeps optional retrieval sources independent."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from hcmai.common.config import AppConfig, IndexConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.llm.config import LLMServiceConfig
from hcmai.orchestration import setup
from hcmai.orchestration.pipeline import SearchService
from hcmai.retriever.pipeline import RetrievalService


class LoadedService:
    def __init__(self, sources) -> None:
        self.active_sources = tuple(sources)


def _settings(tmp_path, available) -> tuple[AppConfig, dict[RetrievalSource, Any]]:
    paths = {
        RetrievalSource.VISUAL: tmp_path / "visual",
        RetrievalSource.CAPTION: tmp_path / "caption",
        RetrievalSource.OCR: tmp_path / "ocr",
        RetrievalSource.ASR: tmp_path / "asr",
    }
    indexes = {}
    for source in available:
        paths[source].mkdir()
        indexes[source] = SimpleNamespace(metadata=SimpleNamespace(
            dataset_version="dataset-v1",
            model_name="google/siglip2-base-patch16-224",
            embedding_dim=8,
        ))
    settings = AppConfig(index=IndexConfig(
        path=paths[RetrievalSource.VISUAL],
        caption_path=paths[RetrievalSource.CAPTION],
        ocr_path=paths[RetrievalSource.OCR],
        asr_path=paths[RetrievalSource.ASR],
    ))
    return settings, indexes


def _load(monkeypatch, tmp_path, available, *, asr_version="dataset-v1"):
    settings, indexes = _settings(tmp_path, available)
    if RetrievalSource.ASR in indexes:
        indexes[RetrievalSource.ASR].metadata.dataset_version = asr_version
    paths = {
        settings.index.path: RetrievalSource.VISUAL,
        settings.index.caption_path: RetrievalSource.CAPTION,
        settings.index.ocr_path: RetrievalSource.OCR,
        settings.index.asr_path: RetrievalSource.ASR,
    }
    monkeypatch.setattr(
        setup.RetrievalService,
        "load_index",
        staticmethod(lambda path, **_: indexes[paths[path]]),
    )
    monkeypatch.setattr(setup, "_query_encoder", lambda *args: object())
    monkeypatch.setattr(
        setup.RetrievalService,
        "from_index",
        staticmethod(lambda *args, **kwargs: LoadedService([RetrievalSource.VISUAL])),
    )
    monkeypatch.setattr(
        setup.RetrievalService,
        "from_indexes",
        staticmethod(
            lambda visual, visual_encoder, text_indexes, *args: LoadedService(
                [RetrievalSource.VISUAL, *text_indexes]
            )
        ),
    )
    for name in (
        "HCMAI_CAPTION_INDEX_PATH",
        "HCMAI_OCR_INDEX_PATH",
        "HCMAI_ASR_INDEX_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    messages = []
    service = setup._load_retrieval(
        settings,
        LLMServiceConfig(),
        settings.index.path,
        None,
        messages,
    )
    assert service is not None
    return cast(LoadedService, service), messages


def test_all_sources_load(monkeypatch, tmp_path) -> None:
    service, messages = _load(monkeypatch, tmp_path, set(RetrievalSource))
    assert service.active_sources == tuple(RetrievalSource)
    assert messages == []


def test_visual_only_is_ready_with_optional_warnings(monkeypatch, tmp_path) -> None:
    service, messages = _load(
        monkeypatch,
        tmp_path,
        {RetrievalSource.VISUAL},
    )
    assert service.active_sources == (RetrievalSource.VISUAL,)
    assert len(messages) == 3


def test_missing_ocr_does_not_disable_other_sources(monkeypatch, tmp_path) -> None:
    service, messages = _load(
        monkeypatch,
        tmp_path,
        {RetrievalSource.VISUAL, RetrievalSource.CAPTION, RetrievalSource.ASR},
    )
    assert RetrievalSource.CAPTION in service.active_sources
    assert RetrievalSource.ASR in service.active_sources
    assert any("OCR index not available" in message for message in messages)


def test_mismatched_asr_is_skipped_independently(monkeypatch, tmp_path) -> None:
    service, messages = _load(
        monkeypatch,
        tmp_path,
        set(RetrievalSource),
        asr_version="wrong-dataset",
    )
    assert service.active_sources == (
        RetrievalSource.VISUAL,
        RetrievalSource.CAPTION,
        RetrievalSource.OCR,
    )
    assert any("dataset version differs" in message for message in messages)


def test_missing_visual_disables_retrieval(monkeypatch, tmp_path) -> None:
    settings, _ = _settings(tmp_path, {RetrievalSource.CAPTION})
    messages = []
    service = setup._load_retrieval(
        settings,
        LLMServiceConfig(),
        settings.index.path,
        None,
        messages,
    )
    assert service is None
    assert any("Index directory not available" in message for message in messages)


def test_health_reports_active_and_inactive_modalities() -> None:
    retrieval = LoadedService(
        [RetrievalSource.VISUAL, RetrievalSource.CAPTION]
    )
    health = SearchService(
        None,
        cast(RetrievalService, retrieval),
    ).health()

    assert health["retrieval_modalities"]["visual"]["active"] is True
    assert health["retrieval_modalities"]["caption"]["active"] is True
    assert health["retrieval_modalities"]["ocr"]["active"] is False
    assert health["retrieval_modalities"]["asr"]["active"] is False
