"""Tests for the modality-neutral exact FAISS dense index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

faiss = pytest.importorskip("faiss")

from hcmai.retriever.dense import DenseIndex, IndexMetadata


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


@pytest.fixture
def corpus():
    """Deterministic normalized embeddings with a matching frame mapping."""
    rng = np.random.default_rng(0)
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
def built_index(corpus):
    embeddings, mapping = corpus
    return DenseIndex.build(embeddings, mapping, dataset_version="test_v1", model_name="google/siglip2-base-patch16-224")


class TestIndexMetadata:
    def test_metadata_roundtrip(self):
        metadata = IndexMetadata(
            dataset_version="v1",
            model_name="m",
            index_type="flat_ip",
            metric="inner_product",
            normalization="l2",
            embedding_dim=8,
            vector_count=10,
            build_time_sec=0.1,
            index_size_bytes=1234,
            generated_at="2026-01-01T00:00:00",
        )
        assert IndexMetadata.from_dict(metadata.to_dict()) == metadata


class TestDenseIndexBuild:
    def test_counts_match_mapping(self, built_index, corpus):
        _, mapping = corpus
        assert built_index.index.ntotal == len(mapping)
        assert built_index.metadata.vector_count == len(mapping)
        assert built_index.metadata.embedding_dim == 8

    def test_rejects_count_mismatch(self, corpus):
        embeddings, mapping = corpus
        with pytest.raises(ValueError, match="does not match"):
            DenseIndex.build(embeddings[:5], mapping, dataset_version="v1", model_name="m")

    def test_rejects_bad_positions(self, corpus):
        embeddings, mapping = corpus
        bad = mapping.copy()
        bad.loc[0, "embedding_index"] = 99
        with pytest.raises(ValueError, match="permutation of 0..N-1"):
            DenseIndex.build(embeddings, bad, dataset_version="v1", model_name="m")

    def test_rejects_duplicate_frame_ids(self, corpus):
        embeddings, mapping = corpus
        bad = mapping.copy()
        bad.loc[1, "frame_id"] = "f000"
        with pytest.raises(ValueError, match="duplicate frame_id"):
            DenseIndex.build(embeddings, bad, dataset_version="v1", model_name="m")


class TestDenseIndexSearch:
    def test_fixture_retrieves_itself(self, built_index, corpus):
        """Each fixture vector must retrieve itself at rank 1."""
        embeddings, mapping = corpus
        scores, positions = built_index.search(embeddings, top_k=1)
        assert positions[:, 0].tolist() == list(range(10))
        # Self inner product of a normalized vector is ~1.
        np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-4)


class TestDenseIndexPersistence:
    def test_save_and_load_roundtrip(self, built_index):
        with TemporaryDirectory() as tmp:
            built_index.save(tmp)
            files = {p.name for p in Path(tmp).iterdir()}
            assert files == {"dense.index", "frame_mapping.parquet", "metadata.json"}

            loaded = DenseIndex.load(tmp)
            assert loaded.index.ntotal == built_index.index.ntotal
            assert loaded.metadata.index_size_bytes > 0

    def test_load_rejects_mismatched_artifacts(self, built_index):
        with TemporaryDirectory() as tmp:
            built_index.save(tmp)
            # Corrupt the mapping so it no longer matches the index count.
            mapping_path = Path(tmp) / "frame_mapping.parquet"
            truncated = pd.read_parquet(mapping_path).iloc[:5]
            truncated.to_parquet(mapping_path)
            with pytest.raises(ValueError, match="Mismatched index artifacts"):
                DenseIndex.load(tmp)
