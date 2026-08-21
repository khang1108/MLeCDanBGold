"""Bounded in-process caches for immutable retrieval artifacts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, Generic, Hashable, Literal, Protocol, TypeVar

import numpy as np

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")
DiskKeyT = TypeVar("DiskKeyT", bound=Hashable, contravariant=True)
SourceFamily = Literal["visual", "text"]


@dataclass(frozen=True, slots=True)
class EmbeddingCacheKey:
    model_name: str
    revision: str | None
    source_family: SourceFamily
    normalized_query: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class ThumbnailCacheKey:
    dataset_version: str
    frame_id: str
    maximum_size: tuple[int, int]
    quality: int


@dataclass(frozen=True, slots=True)
class CacheMetricsSnapshot:
    hits: int
    misses: int
    evictions: int
    entries: int
    bytes_used: int


class DiskCache(Protocol[DiskKeyT, ValueT]):
    """Optional persistent cache boundary; disabled in the baseline."""

    def get(self, key: DiskKeyT) -> ValueT | None: ...

    def set(self, key: DiskKeyT, value: ValueT) -> None: ...


class DisabledDiskCache(Generic[KeyT, ValueT]):
    def get(self, key: KeyT) -> ValueT | None:
        del key
        return None

    def set(self, key: KeyT, value: ValueT) -> None:
        del key, value


@dataclass(slots=True)
class _Entry(Generic[ValueT]):
    value: ValueT
    expires_at: float
    size_bytes: int


class BoundedTTLCache(Generic[KeyT, ValueT]):
    """Thread-safe LRU with TTL, entry count, and byte bounds."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        ttl_seconds: float,
        size_of: Callable[[ValueT], int],
        clock: Callable[[], float] = monotonic,
        disk: DiskCache[KeyT, ValueT] | None = None,
    ) -> None:
        if max_entries < 1 or max_bytes < 1 or ttl_seconds <= 0:
            raise ValueError("cache bounds and TTL must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._size_of = size_of
        self._clock = clock
        self._disk = disk or DisabledDiskCache()
        self._entries: OrderedDict[KeyT, _Entry[ValueT]] = OrderedDict()
        self._bytes_used = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = Lock()

    def get(self, key: KeyT) -> ValueT | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.expires_at > now:
                self._entries.move_to_end(key)
                self._hits += 1
                return entry.value
            if entry is not None:
                self._remove(key, eviction=True)
            self._misses += 1
        disk_value = self._disk.get(key)
        if disk_value is not None:
            self.set(key, disk_value)
        return disk_value

    def set(self, key: KeyT, value: ValueT) -> None:
        size = self._size_of(value)
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._entries:
                self._remove(key, eviction=False)
            self._entries[key] = _Entry(
                value=value,
                expires_at=self._clock() + self.ttl_seconds,
                size_bytes=size,
            )
            self._bytes_used += size
            while (
                len(self._entries) > self.max_entries
                or self._bytes_used > self.max_bytes
            ):
                oldest = next(iter(self._entries))
                self._remove(oldest, eviction=True)
        self._disk.set(key, value)

    def metrics(self) -> CacheMetricsSnapshot:
        with self._lock:
            return CacheMetricsSnapshot(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._entries),
                bytes_used=self._bytes_used,
            )

    def _remove(self, key: KeyT, *, eviction: bool) -> None:
        entry = self._entries.pop(key)
        self._bytes_used -= entry.size_bytes
        if eviction:
            self._evictions += 1


class EmbeddingCache(BoundedTTLCache[EmbeddingCacheKey, np.ndarray]):
    """Store read-only float arrays and never expose mutable cached state."""

    def __init__(self, **kwargs) -> None:
        super().__init__(size_of=lambda value: int(value.nbytes), **kwargs)

    def get(self, key: EmbeddingCacheKey) -> np.ndarray | None:
        value = super().get(key)
        if value is None:
            return None
        view = value.view()
        view.setflags(write=False)
        return view

    def set(self, key: EmbeddingCacheKey, value: np.ndarray) -> None:
        stored = np.array(value, dtype=np.float32, copy=True)
        stored.setflags(write=False)
        super().set(key, stored)


class ThumbnailCache(BoundedTTLCache[ThumbnailCacheKey, bytes]):
    def __init__(self, **kwargs) -> None:
        super().__init__(size_of=len, **kwargs)
