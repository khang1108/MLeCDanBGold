"""Tests for independently reported temporal evidence capabilities."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
from hcmai.common.config import AppConfig
from hcmai.orchestration import setup
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.models import RetrievalSource


class FakeCorpus:
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



class FakeEvidence:
    def __init__(self, dense: bool, bm25: bool) -> None:
        self.dense = object() if dense else None
        self.bm25 = object() if bm25 else None


def test_health_reports_dense_bm25_and_hybrid_independently() -> None:
    corpus = cast(Any, FakeCorpus())
    retrieval = cast(Any, SimpleNamespace(active_sources=()))

    dense_only = SearchService(
        corpus, retrieval, temporal_evidence=cast(Any, FakeEvidence(True, False))
    ).health()["capabilities"]
    bm25_only = SearchService(
        corpus, retrieval, temporal_evidence=cast(Any, FakeEvidence(False, True))
    ).health()["capabilities"]
    both = SearchService(
        corpus, retrieval, temporal_evidence=cast(Any, FakeEvidence(True, True))
    ).health()["capabilities"]

    assert (dense_only["dense_temporal"], dense_only["bm25"], dense_only["hybrid_temporal"]) == (
        True,
        False,
        False,
    )
    assert (bm25_only["dense_temporal"], bm25_only["bm25"], bm25_only["hybrid_temporal"]) == (
        False,
        True,
        False,
    )
    assert both["hybrid_temporal"] is True
    assert both["visual_dense"] is True
    assert both["context_dense"] is True
    assert both["asr_dense"] is True


def test_setup_reuses_dense_bindings_and_loads_bm25_canonical_mapping(
    monkeypatch: Any,
) -> None:
    mapping = pd.DataFrame(
        [{"frame_id": "f1", "video_id": "v1", "frame_idx": 1, "timestamp_ms": 10}]
    )
    visual = SimpleNamespace(index=SimpleNamespace(mapping=mapping), encoder=object())
    context = SimpleNamespace(
        index=SimpleNamespace(metadata=SimpleNamespace(embedding_dim=4)),
        encoder=object(),
    )
    asr_retriever = SimpleNamespace(
        index=SimpleNamespace(metadata=SimpleNamespace(embedding_dim=4)),
        projector=object(),
    )
    dense = object()
    bm25 = object()
    bindings = {
        RetrievalSource.VISUAL: visual,
        RetrievalSource.CONTEXT: context,
        RetrievalSource.ASR: asr_retriever,
    }
    retrieval = SimpleNamespace(source_retriever=bindings.get)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        setup,
        "SegmentProjectedASRIndex",
        lambda **kwargs: captured.setdefault("projected_asr", kwargs) and object(),
        raising=False,
    )
    monkeypatch.setattr(
        setup,
        "DenseTemporalScorer",
        lambda **kwargs: captured.setdefault("dense", kwargs) and dense,
    )
    monkeypatch.setattr(
        setup.BM25TemporalScorer,
        "load",
        lambda path, canonical, weights: captured.setdefault("bm25", (path, canonical, weights))
        and bm25,
    )

    evidence = setup._load_temporal_evidence(AppConfig(), cast(Any, retrieval), [])

    assert evidence is not None
    assert evidence.dense is dense
    assert evidence.bm25 is bm25
    assert captured["dense"]["visual_index"] is visual.index
    assert captured["dense"]["context_index"] is context.index
    assert captured["projected_asr"] == {
        "segment_index": asr_retriever.index,
        "canonical_index": visual.index,
        "projector": asr_retriever.projector,
    }
    assert captured["dense"]["visual_encoder"] is visual.encoder
    assert captured["dense"]["text_encoder"] is context.encoder
    assert captured["dense"]["chunk_size"] == AppConfig().search.alignment.chunk_size
    assert captured["bm25"][1] is mapping


def test_health_reports_partial_dense_source_readiness() -> None:
    evidence = FakeEvidence(False, True)
    evidence.visual_dense_ready = True
    evidence.context_dense_ready = True
    evidence.asr_dense_ready = False
    health = SearchService(
        cast(Any, FakeCorpus()),
        cast(Any, SimpleNamespace(active_sources=())),
        temporal_evidence=cast(Any, evidence),
    ).health()["capabilities"]

    assert health["visual_dense"] is True
    assert health["context_dense"] is True
    assert health["asr_dense"] is False
    assert health["dense_temporal"] is False


def test_dimension_mismatch_produces_truthful_search_health(monkeypatch: Any) -> None:
    """Do not advertise ASR Dense after shared-encoder incompatibility."""

    mapping = pd.DataFrame(
        [{"frame_id": "f1", "video_id": "v1", "frame_idx": 1, "timestamp_ms": 10}]
    )
    bindings = {
        RetrievalSource.VISUAL: SimpleNamespace(index=SimpleNamespace(mapping=mapping)),
        RetrievalSource.CONTEXT: SimpleNamespace(
            index=SimpleNamespace(metadata=SimpleNamespace(embedding_dim=4))
        ),
        RetrievalSource.ASR: SimpleNamespace(
            index=SimpleNamespace(metadata=SimpleNamespace(embedding_dim=8)),
            projector=object(),
        ),
    }
    retrieval = SimpleNamespace(source_retriever=bindings.get, active_sources=())
    monkeypatch.setattr(setup, "_load_bm25_temporal", lambda *args: object())

    evidence = setup._load_temporal_evidence(AppConfig(), cast(Any, retrieval), [])
    assert evidence is not None
    health = SearchService(
        cast(Any, FakeCorpus()),
        cast(Any, retrieval),
        temporal_evidence=evidence,
    ).health()["capabilities"]

    assert health["context_dense"] is True
    assert health["asr_dense"] is False
    assert health["dense_temporal"] is False
