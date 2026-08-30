"""Smoke tests for task-agnostic reciprocal-rank fusion."""

from __future__ import annotations

import numpy as np

from hcmai.common.config import EncoderConfig, FusionConfig
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
)
from hcmai.retrieval.retriever.fusion import RRFFusionRetriever
from hcmai.retrieval.retriever.query_batch import encode_query_batch


class _FakeEncoder:
    """Return one normalized vector for each requested query."""

    def __init__(self, model_name: str) -> None:
        self.config = EncoderConfig(model_name=model_name)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode fixture text without model or network dependencies."""

        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _FakeRetriever:
    """Return one source-local ranking over canonical fixture frame IDs."""

    def __init__(self, source: RetrievalSource, frame_ids: list[str]) -> None:
        self.source = source
        self.source_family = "visual" if source is RetrievalSource.VISUAL else "text"
        self.encoder = _FakeEncoder(f"fixture/{self.source_family}")
        self.frame_ids = frame_ids

    def encode(self, queries: list[str]):
        """Build the normal ordered query batch consumed by concurrent fusion."""

        return encode_query_batch(queries, self.encoder, self.source_family)

    def search_vectors(self, query_batch, top_k: int = 100) -> list[RetrievalResult]:
        """Return the same deterministic source ranking for every query."""

        candidates = [
            RetrievalCandidate(
                frame_id=frame_id,
                source_scores={self.source: 1.0 / rank},
                source_ranks={self.source: rank},
            )
            for rank, frame_id in enumerate(self.frame_ids[:top_k], start=1)
        ]
        return [
            RetrievalResult(candidates=candidates)
            for _ in query_batch.embeddings
        ]


def test_equal_default_source_weights_produce_equal_weight_rrf_order() -> None:
    """Fuse sources without a task or filter argument and keep RRF ordering."""

    fusion = RRFFusionRetriever(
        [
            _FakeRetriever(RetrievalSource.VISUAL, ["shared", "visual-only"]),
            _FakeRetriever(RetrievalSource.CONTEXT, ["context-only", "shared"]),
        ],
        FusionConfig(
            required_sources={RetrievalSource.VISUAL, RetrievalSource.CONTEXT}
        ),
    )

    result = fusion.search("query", top_k=10)

    assert [candidate.frame_id for candidate in result] == [
        "shared",
        "context-only",
        "visual-only",
    ]
    assert result[0].fusion_score == (1 / 61) + (1 / 62)
