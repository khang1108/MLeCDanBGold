"""Chunked subset scoring must match a plain dense matmul."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.retriever.dense.index import DenseIndex  # noqa: E402


def test_score_subset_matches_matmul_across_chunks() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    mapping = pd.DataFrame(
        {
            "embedding_index": range(20),
            "frame_id": [f"f{i}" for i in range(20)],
            "video_id": ["v"] * 20,
            "frame_idx": range(20),
            "timestamp_ms": range(20),
        }
    )
    index = DenseIndex.build(embeddings, mapping, dataset_version="t", model_name="t")

    queries = embeddings[[3, 11]]
    positions = np.array([17, 2, 9, 9, 0], dtype=np.int64)
    scores = index.score_subset(queries, positions, chunk_size=2)

    assert scores.shape == (2, 5)
    assert np.allclose(scores, queries @ embeddings[positions].T, atol=1e-6)
