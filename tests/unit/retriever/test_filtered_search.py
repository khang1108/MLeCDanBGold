"""Exact candidate-local vector filtering regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.common.schemas.search import SearchFilters
from hcmai.retrieval.retriever.dense.index import DenseIndex


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


@pytest.fixture
def corpus():
    rng = np.random.default_rng(42)
    vectors = _normalize(rng.normal(size=(120, 16)).astype(np.float32))
    mapping = pd.DataFrame({
        "frame_id": [f"frame-{index}" for index in range(120)],
        "video_id": [f"video-{index // 10:02d}" for index in range(120)],
        "frame_idx": list(range(120)),
        "timestamp_ms": [(index % 10) * 1000 for index in range(120)],
        "embedding_index": list(range(120)),
    })
    return vectors, DenseIndex.build(
        vectors,
        mapping,
        dataset_version="fixture-v1",
        model_name="fixture/model",
    )


@pytest.mark.parametrize(
    "video_ids",
    [["video-03"], [f"video-{index:02d}" for index in range(10)]],
)
def test_filtered_results_match_exhaustive_faiss_postfilter(corpus, video_ids) -> None:
    vectors, index = corpus
    query = vectors[[37]]
    filters = SearchFilters(video_ids=video_ids)
    allowed = set(index.filtered_positions(filters).tolist())

    full_scores, full_positions = index.search(query, index.index.ntotal)
    oracle = [
        (float(score), int(position))
        for score, position in zip(full_scores[0], full_positions[0])
        if int(position) in allowed
    ][:7]
    scores, positions = index.search_filtered(query, 7, filters)

    assert positions[0].tolist() == [position for _, position in oracle]
    np.testing.assert_allclose(scores[0], [score for score, _ in oracle], atol=1e-6)


def test_video_and_time_filter_searches_only_resolved_subset(corpus) -> None:
    vectors, index = corpus

    class SearchSpy:
        def search(self, *_):
            raise AssertionError("filtered search must not call FAISS")

    index.index = SearchSpy()
    filters = SearchFilters(
        video_ids=["video-04"],
        start_time_ms=2000,
        end_time_ms=5000,
    )
    _, positions = index.search_filtered(vectors[[45]], 10, filters)

    assert positions[0].tolist()[0] == 45
    assert set(positions[0]).issubset({42, 43, 44, 45})


def test_empty_filtered_subset_returns_empty_batch(corpus) -> None:
    vectors, index = corpus
    scores, positions = index.search_filtered(
        vectors[:2],
        10,
        SearchFilters(video_ids=["missing-video"]),
    )

    assert scores.shape == positions.shape == (2, 0)


def test_unrestricted_search_continues_to_use_faiss(corpus) -> None:
    vectors, index = corpus
    calls = []
    original = index.index

    class SearchSpy:
        ntotal = original.ntotal

        def search(self, queries, top_k):
            calls.append((len(queries), top_k))
            return original.search(queries, top_k)

    index.index = SearchSpy()
    index.search_filtered(vectors[:2], 5, None)

    assert calls == [(2, 5)]
