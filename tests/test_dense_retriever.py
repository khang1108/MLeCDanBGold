"""Tests for the online dense retriever contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

faiss = pytest.importorskip("faiss")

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.models import RetrievalCandidate, RetrievalSource
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever

MODEL_NAME = "google/siglip2-base-patch16-224"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


class FakeEncoder:
    """Stand-in encoder that returns a chosen corpus vector for any query."""

    def __init__(self, embeddings: np.ndarray, model_name: str, target: int) -> None:
        self.config = EncoderConfig(model_name=model_name)
        self.embedding_dim = embeddings.shape[1]
        self._embeddings = embeddings
        self.target = target

    def encode_text(self, texts, stats=None) -> np.ndarray:
        return self._embeddings[self.target : self.target + 1]


@pytest.fixture
def corpus():
    rng = np.random.default_rng(1)
    embeddings = _normalize(rng.standard_normal((10, 8)).astype(np.float32))
    mapping = pd.DataFrame(
        {
            "frame_id": [f"f{i:03d}" for i in range(10)],
            "video_id": ["v001"] * 5 + ["v002"] * 5,
            "frame_idx": list(range(10)),
            "embedding_index": list(range(10)),
            "timestamp_ms": [i * 1000 for i in range(10)],
        }
    )
    return embeddings, mapping


@pytest.fixture
def index(corpus):
    embeddings, mapping = corpus
    return DenseIndex.build(embeddings, mapping, dataset_version="test_v1", model_name=MODEL_NAME)


class TestConstruction:
    def test_rejects_model_mismatch(self, corpus, index):
        embeddings, _ = corpus
        encoder = FakeEncoder(embeddings, "other/model", target=0)
        with pytest.raises(ValueError, match="model mismatch"):
            DenseRetriever(encoder, index)


class TestSearch:
    def test_target_frame_ranks_first(self, corpus, index):
        embeddings, _ = corpus
        retriever = DenseRetriever(FakeEncoder(embeddings, MODEL_NAME, 3), index)
        candidates = retriever.search("a query", top_k=5)

        assert candidates[0].frame_id == "f003"
        assert candidates[0].source_ranks[RetrievalSource.VISUAL] == 1
        assert candidates[0].source_scores[RetrievalSource.VISUAL] == pytest.approx(
            1.0, abs=1e-4
        )

    def test_returns_candidate_contract(self, corpus, index):
        embeddings, _ = corpus
        retriever = DenseRetriever(FakeEncoder(embeddings, MODEL_NAME, 0), index)
        candidates = retriever.search("q", top_k=5)

        assert all(isinstance(c, RetrievalCandidate) for c in candidates)
        assert candidates[0].metadata["frame"]["video_id"] == "v001"

    def test_respects_top_k(self, corpus, index):
        embeddings, _ = corpus
        retriever = DenseRetriever(FakeEncoder(embeddings, MODEL_NAME, 0), index)
        assert len(retriever.search("q", top_k=3)) == 3

    def test_no_duplicate_frame_ids(self, corpus, index):
        embeddings, _ = corpus
        retriever = DenseRetriever(FakeEncoder(embeddings, MODEL_NAME, 0), index)
        ids = [c.frame_id for c in retriever.search("q", top_k=10)]
        assert len(ids) == len(set(ids))

    def test_records_latency_separately(self, corpus, index):
        embeddings, _ = corpus
        retriever = DenseRetriever(FakeEncoder(embeddings, MODEL_NAME, 0), index)
        result = retriever.search("q", top_k=5)
        assert result.trace.duration_for("query_encoding") >= 0.0
        assert result.trace.duration_for("index_search") >= 0.0
