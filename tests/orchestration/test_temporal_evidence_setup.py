"""Tests for temporal evidence setup and independent capability loading."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pandas as pd

from hcmai.common.config import AppConfig
from hcmai.orchestration.setup import _load_dense_temporal
from hcmai.retrieval.models import RetrievalSource
from tests.retrieval.evidence.fakes import FakeEncoder, FakeIndex


class FakeSourceRetriever:
    def __init__(self, source, index, encoder) -> None:
        self.source = source
        self.index = index
        self.encoder = encoder


class FakeRetrievalService:
    def __init__(self, retrievers) -> None:
        self.retrievers = {retriever.source: retriever for retriever in retrievers}

    def source_retriever(self, source):
        return self.retrievers.get(source)


class FakeSegmentIndex:
    def __init__(self) -> None:
        self.mapping = pd.DataFrame(
            [{"video_id": "v1", "start_ms": 0, "end_ms": 1000}]
        )
        self.vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
        self.metadata = SimpleNamespace(embedding_dim=2)


class FakeProjection:
    video_id = "v1"
    frame_id = "f0"
    frame_idx = 0
    timestamp_ms = 0


class FakeProjector:
    def project(self, video_id: str, *, start_ms: int, end_ms: int):
        del video_id, start_ms, end_ms
        return FakeProjection()


class FakeASRRetriever(FakeSourceRetriever):
    def __init__(self, encoder) -> None:
        super().__init__(RetrievalSource.ASR, FakeSegmentIndex(), encoder)
        self.projector = FakeProjector()


def test_dense_loader_keeps_visual_context_when_asr_is_missing() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    context = FakeSourceRetriever(RetrievalSource.CONTEXT, FakeIndex(scores), encoder)
    retrieval = FakeRetrievalService([visual, context])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(), retrieval, visual, messages := []
    )

    assert scorer is not None
    assert context_ready is True
    assert asr_ready is False
    assert any("ASR segment retriever missing" in message for message in messages)


def test_dense_loader_keeps_visual_asr_when_context_is_missing() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    asr = FakeASRRetriever(encoder)
    retrieval = FakeRetrievalService([visual, asr])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(), retrieval, visual, messages := []
    )

    assert scorer is not None
    assert context_ready is False
    assert asr_ready is True
