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
    SearchRequest,
    StageStatus,
    StageTrace,
)
from hcmai.orchestration.ranking import rank_candidates
from hcmai.retriever.dense.index import DenseIndex
from hcmai.retriever.dense.retriever import DenseRetriever
from hcmai.retriever.pipeline import RetrievalService

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


class TracedRetrieval:
    def search(self, query, top_k, filters, query_type):
        del query, top_k, filters, query_type
        return RetrievalResult(
            candidates=[
                RetrievalCandidate(
                    frame_id="frame-1",
                    source_scores={RetrievalSource.VISUAL: 0.9},
                    source_ranks={RetrievalSource.VISUAL: 1},
                )
            ],
            trace=RetrievalTrace(
                stages={
                    "visual.query_encoding": _stage(
                        "visual.query_encoding", 2
                    ),
                    "visual.index_search": _stage("visual.index_search", 3),
                }
            ),
        )


def test_orchestration_emits_required_json_stage_fields(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="hcmai.orchestration.ranking"):
        result, reranking_ms = rank_candidates(
            SearchRequest(query="red bus"),
            cast(RetrievalService, TracedRetrieval()),
            None,
            candidate_count=20,
            rerank_count=0,
            request_id="request-1",
        )

    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith("{")
    ]
    assert result.candidates[0].frame_id == "frame-1"
    assert reranking_ms == 0
    assert records
    assert all(
        {"request_id", "task_type", "stage", "duration_ms", "status"}
        <= record.keys()
        for record in records
    )
    assert {record["stage"] for record in records} == {
        "visual.query_encoding",
        "visual.index_search",
        "rerank",
    }
