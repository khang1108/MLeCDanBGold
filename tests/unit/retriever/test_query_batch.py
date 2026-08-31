"""Batch-query encoding and vector reuse regression tests."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter, sleep

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.models import RetrievalSource
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever
from hcmai.retrieval.retriever.query_batch import encode_query_batch


class CountingEncoder:
    def __init__(self, model_name: str, delay_seconds: float = 0.0) -> None:
        self.config = EncoderConfig(model_name=model_name)
        self.embedding_dim = 3
        self.resolved_revision = "fixture-r1"
        self.delay_seconds = delay_seconds
        self.calls: list[list[str]] = []

    def encode_text(self, texts, stats=None) -> np.ndarray:
        del stats
        self.calls.append(list(texts))
        if self.delay_seconds:
            sleep(self.delay_seconds)
        vectors = np.eye(3, dtype=np.float32)
        return np.stack(
            [vectors[sum(text.encode("utf-8")) % len(vectors)] for text in texts]
        )


def _index(model_name: str) -> DenseIndex:
    mapping = pd.DataFrame(
        {
            "frame_id": ["f0", "f1", "f2"],
            "video_id": ["v0", "v0", "v1"],
            "frame_idx": [0, 1, 2],
            "timestamp_ms": [0, 1000, 2000],
            "embedding_index": [0, 1, 2],
        }
    )
    return DenseIndex.build(
        np.eye(3, dtype=np.float32),
        mapping,
        dataset_version="fixture-v1",
        model_name=model_name,
    )


def test_query_batch_deduplicates_normalized_text_and_restores_order() -> None:
    encoder = CountingEncoder("fixture/text")

    batch = encode_query_batch([" event  one ", "event one", "event two"], encoder, "text")

    assert encoder.calls == [["event one", "event two"]]
    assert [item.query.text for item in batch.embeddings] == [
        " event  one ",
        "event one",
        "event two",
    ]
    assert [item.query.position for item in batch.embeddings] == [0, 1, 2]
    assert batch.revision == "fixture-r1"
    assert batch.model_name == "fixture/text"
    assert batch.source_family == "text"


def test_search_vectors_rejects_incompatible_batch_provenance() -> None:
    encoder = CountingEncoder("fixture/visual")
    retriever = DenseRetriever(encoder, _index("fixture/visual"))
    batch = encode_query_batch(["query"], encoder, "visual")
    embedding = batch.embeddings[0]

    wrong_model = replace(
        batch,
        embeddings=(replace(embedding, model_name="other/model"),),
    )
    wrong_dimension = replace(
        batch,
        embeddings=(replace(embedding, vector=np.array([1.0, 0.0])),),
    )
    wrong_family = replace(
        batch,
        embeddings=(
            replace(embedding, query=replace(embedding.query, source_family="text")),
        ),
    )
    unnormalized = replace(
        batch,
        embeddings=(replace(embedding, vector=np.array([2.0, 0.0, 0.0])),),
    )

    with pytest.raises(ValueError, match="model names"):
        retriever.search_vectors(wrong_model)
    with pytest.raises(ValueError, match="dimensions"):
        retriever.search_vectors(wrong_dimension)
    with pytest.raises(ValueError, match="source family"):
        retriever.search_vectors(wrong_family)
    with pytest.raises(ValueError, match="L2-normalized"):
        retriever.search_vectors(unnormalized)


def test_batch_reuse_reduces_encoder_wait_on_small_fixture() -> None:
    encoder = CountingEncoder("fixture/text", delay_seconds=0.02)
    retrievers = [
        DenseRetriever(encoder, _index("fixture/text"), source)
        for source in (
            RetrievalSource.CAPTION,
            RetrievalSource.OCR,
            RetrievalSource.ASR,
        )
    ]

    started = perf_counter()
    for retriever in retrievers:
        retriever.search("event", top_k=1)
    independent_elapsed = perf_counter() - started

    encoder.calls.clear()
    batch = retrievers[0].encode(["event"])
    started = perf_counter()
    for retriever in retrievers:
        retriever.search_vectors(batch, top_k=1)
    reused_elapsed = perf_counter() - started + encoder.delay_seconds

    assert encoder.calls == [["event"]]
    assert reused_elapsed < independent_elapsed
