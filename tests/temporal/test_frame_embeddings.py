"""Tests for frame embedding accessor (SP-04).

Verifies Task 3.1:
- Do not decode and re-encode the source video.
- Retrieve embeddings already stored in the visual index.
- Preserve canonical frame indexing.
- The same candidate frame always resolves to the same stored visual vector.
- Handle empty indices, out-of-bounds indices, and various backend sources.
"""

import numpy as np
import pandas as pd
import pytest

from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.models.metadata import IndexMetadata
from hcmai.temporal.transition_decoder import FrameEmbeddingAccessor


class MockFaissIndex:
    """Mock FAISS index flat IP."""

    def __init__(self, d: int, ntotal: int):
        self.d = d
        self.ntotal = ntotal


def _create_test_dense_index() -> DenseIndex:
    dim = 8
    # 2 videos:
    # video-A has 3 frames: frame_idx [0, 2, 5], mapped to embedding_indices [0, 1, 2]
    # video-B has 2 frames: frame_idx [1, 3], mapped to embedding_indices [3, 4]
    vectors = np.random.default_rng(42).normal(size=(5, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    mapping = pd.DataFrame([
        {"embedding_index": 0, "video_id": "video-A", "frame_id": "vA_0", "frame_idx": 0, "timestamp_ms": 1000},
        {"embedding_index": 1, "video_id": "video-A", "frame_id": "vA_2", "frame_idx": 2, "timestamp_ms": 3000},
        {"embedding_index": 2, "video_id": "video-A", "frame_id": "vA_5", "frame_idx": 5, "timestamp_ms": 6000},
        {"embedding_index": 3, "video_id": "video-B", "frame_id": "vB_1", "frame_idx": 1, "timestamp_ms": 2000},
        {"embedding_index": 4, "video_id": "video-B", "frame_id": "vB_3", "frame_idx": 3, "timestamp_ms": 4000},
    ])

    metadata = IndexMetadata(
        dataset_version="test-v1",
        model_name="mock-model",
        index_type="flat_ip",
        metric="inner_product",
        normalization="l2",
        embedding_dim=dim,
        vector_count=5,
        build_time_sec=1.0,
        index_size_bytes=1000,
        generated_at="2026-01-01T00:00:00Z",
    )

    return DenseIndex(
        index=MockFaissIndex(dim, 5),
        mapping=mapping,
        metadata=metadata,
        vectors=vectors,
    )


def test_dense_index_get_frame_embeddings():
    """Verify DenseIndex.get_frame_embeddings retrieves canonical candidate vectors."""
    index = _create_test_dense_index()

    # video-A has 3 frames in canonical order: pos 0 (frame_idx 0), pos 1 (frame_idx 2), pos 2 (frame_idx 5)
    # Request candidates at pos 0 and 2
    embs = index.get_frame_embeddings("video-A", [0, 2])
    assert embs.shape == (2, 8)
    assert embs.dtype == np.float32

    # Verify identical to vectors at embedding_indices 0 and 2
    np.testing.assert_array_equal(embs[0], index.vectors[0])
    np.testing.assert_array_equal(embs[1], index.vectors[2])

    # Idempotent: repeated access gives identical vectors
    embs_again = index.get_frame_embeddings("video-A", [0, 2])
    np.testing.assert_array_equal(embs, embs_again)

    # Empty indices
    empty = index.get_frame_embeddings("video-A", [])
    assert empty.shape == (0, 8)

    # Out of bounds raises IndexError
    with pytest.raises(IndexError):
        index.get_frame_embeddings("video-A", [3])  # video-A only has 3 frames (0..2)
    with pytest.raises(IndexError):
        index.get_frame_embeddings("video-A", [-1])


def test_frame_embedding_accessor_wrapping_dense_index():
    """Verify FrameEmbeddingAccessor works with DenseIndex."""
    index = _create_test_dense_index()
    accessor = FrameEmbeddingAccessor(index)

    embs = accessor("video-B", np.array([1, 0]))
    assert embs.shape == (2, 8)
    # video-B: pos 0 is frame_idx 1 (vector 3), pos 1 is frame_idx 3 (vector 4)
    np.testing.assert_array_equal(embs[0], index.vectors[4])
    np.testing.assert_array_equal(embs[1], index.vectors[3])


def test_frame_embedding_accessor_wrapping_dict():
    """Verify FrameEmbeddingAccessor works with per-video embedding arrays."""
    rng = np.random.default_rng(123)
    dict_source = {
        "vid_1": rng.normal(size=(4, 16)).astype(np.float32),
        "vid_2": rng.normal(size=(6, 16)).astype(np.float32),
    }
    accessor = FrameEmbeddingAccessor(dict_source)

    embs = accessor("vid_1", [1, 3])
    assert embs.shape == (2, 16)
    np.testing.assert_array_equal(embs, dict_source["vid_1"][[1, 3]])

    # Unknown video raises KeyError
    with pytest.raises(KeyError):
        accessor("vid_unknown", [0])

    # Out of bounds raises IndexError
    with pytest.raises(IndexError):
        accessor("vid_1", [10])
