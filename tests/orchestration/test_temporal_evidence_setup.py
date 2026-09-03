"""Characterization tests for temporal evidence setup and loading."""

from __future__ import annotations

import numpy as np

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


def test_v9_dense_loader_requires_context_and_asr() -> None:
    scores = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    encoder = FakeEncoder(np.asarray([[1.0, 0.0]], dtype=np.float32))
    visual = FakeSourceRetriever(RetrievalSource.VISUAL, FakeIndex(scores), encoder)
    context = FakeSourceRetriever(RetrievalSource.CONTEXT, FakeIndex(scores), encoder)
    retrieval = FakeRetrievalService([visual, context])

    scorer, context_ready, asr_ready = _load_dense_temporal(
        AppConfig(),
        retrieval,
        visual,
        messages := [],
    )

    assert scorer is None
    assert context_ready is True
    assert asr_ready is False
    assert any("ASR segment retriever missing" in message for message in messages)
