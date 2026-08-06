"""Embedding and thumbnail cache correctness tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.image import thumbnail_jpeg_bytes
from hcmai.retriever.cache import (
    BoundedTTLCache,
    EmbeddingCache,
    ThumbnailCache,
    ThumbnailCacheKey,
)
from hcmai.retriever.query_batch import encode_query_batch


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Encoder:
    def __init__(self, revision: str = "r1") -> None:
        self.config = EncoderConfig(model_name="fixture/model")
        self.embedding_dim = 2
        self.resolved_revision = revision
        self.calls: list[list[str]] = []

    def encode_text(self, texts, stats=None):
        del stats
        self.calls.append(list(texts))
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _embedding_cache(clock=None, max_entries=4, max_bytes=1024):
    return EmbeddingCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
        ttl_seconds=10,
        clock=clock or FakeClock(),
    )


def test_repeated_query_hits_cache_and_returns_read_only_vector() -> None:
    encoder = Encoder()
    cache = _embedding_cache()

    first = encode_query_batch(["  red   bus "], encoder, "text", cache, "p1")
    second = encode_query_batch(["red bus"], encoder, "text", cache, "p1")

    assert encoder.calls == [["red bus"]]
    assert first.encoding_trace.cache_hit is False
    assert second.encoding_trace.cache_hit is True
    assert second.embeddings[0].vector.flags.writeable is False
    metrics = cache.metrics()
    assert (metrics.hits, metrics.misses) == (1, 1)


def test_revision_and_prompt_version_naturally_invalidate_embedding() -> None:
    encoder = Encoder("r1")
    cache = _embedding_cache()
    encode_query_batch(["query"], encoder, "visual", cache, "p1")

    encoder.resolved_revision = "r2"
    encode_query_batch(["query"], encoder, "visual", cache, "p1")
    encode_query_batch(["query"], encoder, "visual", cache, "p2")

    assert encoder.calls == [["query"], ["query"], ["query"]]


def test_ttl_and_lru_memory_bounds_record_evictions() -> None:
    clock = FakeClock()
    cache = BoundedTTLCache[str, bytes](
        max_entries=2,
        max_bytes=4,
        ttl_seconds=5,
        size_of=len,
        clock=clock,
    )
    cache.set("a", b"aa")
    cache.set("b", b"bb")
    assert cache.get("a") == b"aa"
    cache.set("c", b"cc")
    assert cache.get("b") is None
    assert cache.metrics().evictions == 1

    clock.now = 6
    assert cache.get("a") is None
    assert cache.metrics().evictions == 2


def test_cache_operations_are_thread_safe() -> None:
    cache = BoundedTTLCache[int, bytes](
        max_entries=16,
        max_bytes=1024,
        ttl_seconds=60,
        size_of=len,
    )

    def use_cache(value: int) -> None:
        key = value % 16
        cache.set(key, bytes([key]))
        assert cache.get(key) == bytes([key])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(use_cache, range(500)))

    metrics = cache.metrics()
    assert metrics.entries <= 16
    assert metrics.bytes_used <= 1024


def test_thumbnail_cache_uses_compressed_bytes_and_canonical_key(tmp_path) -> None:
    path = tmp_path / "frame.png"
    Image.new("RGB", (64, 32), (255, 0, 0)).save(path)
    cache = ThumbnailCache(
        max_entries=4,
        max_bytes=1024 * 1024,
        ttl_seconds=60,
    )
    key = ThumbnailCacheKey("dataset-v1", "frame-1", (16, 16), 80)

    first = thumbnail_jpeg_bytes(
        path,
        key=key,
        cache=cache,
        maximum_size=key.maximum_size,
        quality=key.quality,
    )
    path.unlink()
    second = thumbnail_jpeg_bytes(
        path,
        key=key,
        cache=cache,
        maximum_size=key.maximum_size,
        quality=key.quality,
    )

    assert first == second
    assert first.startswith(b"\xff\xd8")
    assert cache.metrics().hits == 1
