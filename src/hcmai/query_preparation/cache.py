"""Bounded process-local cache for validated query-preparation results.

Cache identity preserves event case and model lineage. This module does not
perform inference or validate generated event bundles.
"""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from threading import RLock
from time import monotonic
from typing import Any

CacheKey = tuple[str, ...]


def cache_key(
    *,
    operation: str,
    events: Sequence[str],
    model_name: str,
    model_revision: str,
    prompt_version: str,
) -> CacheKey:
    """Build a deterministic key while preserving case-sensitive tokens."""

    normalized_events = tuple(" ".join(event.split()) for event in events)
    return (
        operation,
        model_name,
        model_revision,
        prompt_version,
        *normalized_events,
    )


class QueryPreparationCache:
    """Store recent validated results with TTL and LRU-style eviction."""

    def __init__(
        self,
        *,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        """Initialize cache bounds and an injectable monotonic clock."""

        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")

        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Hashable) -> Any | None:
        """Return an unexpired value and mark it as recently used."""

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if expires_at <= self._clock():
                del self._entries[key]
                return None

            self._entries.move_to_end(key)
            return value

    def put(self, key: Hashable, value: Any) -> None:
        """Store a value and evict least-recently-used entries as needed."""

        with self._lock:
            self._entries[key] = (self._clock() + self._ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)