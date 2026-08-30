from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import cast

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import (
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    RetrievalTrace,
    StageStatus,
    StageTrace,
)
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever
from hcmai.retrieval.retriever.pipeline import RetrievalService

MODEL_NAME = "test/concurrent-encoder"


def _stage(stage: str, duration_ms: float) -> StageTrace:
    return StageTrace(
        stage=stage,
        started_at=1.0,
        ended_at=1.0 + duration_ms / 1_000,
        duration_ms=duration_ms,
        status=StageStatus.SUCCESS,
    )


class ConcurrentEncoder:
    """Synchronize two calls and make their encoding durations distinguishable."""

    def __init__(self, barrier: Barrier) -> None:
        self.config = EncoderConfig(model_name=MODEL_NAME)
        self.embedding_dim = 2
        self.barrier = barrier

    def encode_text(self, texts, stats=None) -> np.ndarray:
        del stats
        self.barrier.wait(timeout=2)
        if texts[0] == "slow":
            time.sleep(0.05)
            return np.asarray([[1.0, 0.0]], dtype=np.float32)
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


def _dense_retriever() -> DenseRetriever:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mapping = pd.DataFrame(
        {
            "frame_id": ["frame-slow", "frame-fast"],
            "video_id": ["video-1", "video-1"],
            "frame_idx": [10, 20],
            "embedding_index": [0, 1],
            "timestamp_ms": [1_000, 2_000],
        }
    )
    index = DenseIndex.build(
        embeddings,
        mapping,
        dataset_version="trace-test",
        model_name=MODEL_NAME,
    )
    return DenseRetriever(ConcurrentEncoder(Barrier(2)), index)


def test_parallel_calls_on_one_retriever_own_independent_traces() -> None:
    retriever = _dense_retriever()

    with ThreadPoolExecutor(max_workers=2) as executor:
        slow_future = executor.submit(retriever.search, "slow", 1)
        fast_future = executor.submit(retriever.search, "fast", 1)
        slow = slow_future.result(timeout=2)
        fast = fast_future.result(timeout=2)

    assert not hasattr(retriever, "last_query_encoding_ms")
    assert not hasattr(retriever, "last_index_search_ms")
    assert slow.trace is not fast.trace
    assert slow.trace.stages["encode"] is not fast.trace.stages[
        "encode"
    ]
    assert slow.trace.duration_for("query_encoding") > (
        fast.trace.duration_for("query_encoding") + 30
    )
    assert slow.candidates[0].frame_id == "frame-slow"
    assert fast.candidates[0].frame_id == "frame-fast"
