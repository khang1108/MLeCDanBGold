"""Tests for query event encoding helper (SP-05).

Verifies Task 3.2:
- Encode query events in the shared vision-language space (SigLIP).
- Output Q = [q_1, ..., q_M] with shape (M, D).
- Encode all event texts once per query (single batch call).
- Output vectors are strictly L2-normalized.
"""

import numpy as np
import pytest

from hcmai.temporal.transition_decoder import encode_query_events


class MockTextEncoder:
    """Mock text encoder tracking call count and batch size."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.call_count = 0
        self.last_batch: list[str] | None = None

    def encode_text(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.last_batch = texts
        # Return unnormalized vectors with random magnitude to test normalization
        rng = np.random.default_rng(len(texts))
        raw = rng.normal(loc=1.0, scale=0.5, size=(len(texts), self.dim)).astype(np.float32)
        return raw * 5.0


def test_encode_query_events_single_batch():
    """Verify all event texts are encoded once per query in a single batch."""
    encoder = MockTextEncoder(dim=12)
    events = ["person sitting down", "person reaches for cup", "person drinks tea"]

    q = encode_query_events(events, encoder=encoder)

    assert encoder.call_count == 1
    assert encoder.last_batch == events
    assert q.shape == (3, 12)
    assert q.dtype == np.float32

    # Verify L2 normalization
    norms = np.linalg.norm(q, axis=-1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-5)


def test_encode_query_events_empty():
    """Verify empty event sequence returns empty array."""
    q = encode_query_events([])
    assert q.shape == (0, 0)


def test_encode_query_events_text_embedding_source():
    """Verify encode_query_events works with an object implementing TextEmbeddingSource."""
    class SimpleTextEncoder:
        def encode_text(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 4), dtype=np.float32)

    q = encode_query_events(["e1", "e2"], encoder=SimpleTextEncoder())
    assert q.shape == (2, 4)
    # L2 normalized from [1, 1, 1, 1] is [0.5, 0.5, 0.5, 0.5]
    np.testing.assert_allclose(q, np.full((2, 4), 0.5))


def test_encode_query_events_rejects_plain_callable():
    """Verify plain callable without encode_text raises TypeError."""
    def simple_fn(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)

    with pytest.raises(TypeError, match="encoder must implement TextEmbeddingSource"):
        encode_query_events(["e1", "e2"], encoder=simple_fn)


def test_encode_query_events_shape_mismatch_raises():
    """Verify invalid encoder output shape raises ValueError."""
    class BadTextEncoder:
        def encode_text(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts) + 1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected encoder to return shape"):
        encode_query_events(["e1"], encoder=BadTextEncoder())
