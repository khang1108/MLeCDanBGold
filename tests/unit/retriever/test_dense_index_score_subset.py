"""Chunked subset scoring must match a plain dense matmul."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.retrieval.retriever.dense.index import (  # noqa: E402
    VECTORS_FILENAME,
    DenseIndex,
    IndexArtifactError,
)


def _index() -> tuple[DenseIndex, np.ndarray]:
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
    built = DenseIndex.build(embeddings, mapping, dataset_version="t", model_name="t")
    return built, embeddings


def test_score_subset_matches_matmul_across_chunks() -> None:
    index, embeddings = _index()

    queries = embeddings[[3, 11]]
    positions = np.array([17, 2, 9, 9, 0], dtype=np.int64)
    scores = index.score_subset(queries, positions, chunk_size=2)

    assert scores.shape == (2, 5)
    assert np.allclose(scores, queries @ embeddings[positions].T, atol=1e-6)


def test_score_subset_reads_an_ascending_run_as_a_slice() -> None:
    index, embeddings = _index()

    queries = embeddings[[3, 11]]
    run = np.arange(4, 18, dtype=np.int64)

    scores = index.score_subset(queries, run, chunk_size=5)
    gathered = index.score_subset(queries, run[::-1].copy(), chunk_size=5)

    assert np.array_equal(scores, queries @ embeddings[run].T)
    assert np.array_equal(scores, gathered[:, ::-1])


def test_an_incomplete_index_bundle_is_rejected_without_runtime_backfill(
    tmp_path,
) -> None:
    index, _ = _index()
    index_dir = index.save(tmp_path / "visual")
    (index_dir / VECTORS_FILENAME).unlink()
    files_before = sorted(path.name for path in index_dir.iterdir())

    with pytest.raises(IndexArtifactError, match=r"missing vectors\.npy"):
        DenseIndex.load(index_dir)

    assert sorted(path.name for path in index_dir.iterdir()) == files_before


def test_a_complete_index_bundle_loads_vectors_read_only(tmp_path) -> None:
    index, embeddings = _index()
    index_dir = index.save(tmp_path / "visual")

    loaded = DenseIndex.load(index_dir)

    assert isinstance(loaded.vectors, np.memmap)
    assert loaded.vectors.mode == "r"
    assert np.allclose(loaded.vectors, embeddings, atol=1e-6)


def test_vector_shape_mismatch_is_rejected(tmp_path) -> None:
    index, _ = _index()
    index_dir = index.save(tmp_path / "visual")
    np.save(index_dir / VECTORS_FILENAME, np.zeros((19, 8), dtype=np.float32))

    with pytest.raises(IndexArtifactError, match="vectors do not match"):
        DenseIndex.load(index_dir)
