"""Fixed-TTL and bounded LRU stores for EventTrail snapshots."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import time

from hcmai.event_trail.errors import EventTrailError
from hcmai.event_trail.models import EvidenceSnapshot


class EvidenceSnapshotStore:
    """Bounded, fixed-TTL store for immutable evidence snapshots.

    LRU order is updated on get/put, but expiration is fixed (not sliding).
    Time-expired entries are kept in a bounded tombstone set so callers
    receive SNAPSHOT_EXPIRED rather than SNAPSHOT_NOT_FOUND. Capacity-evicted
    entries are not tombstoned and return SNAPSHOT_NOT_FOUND.
    """

    def __init__(
        self,
        ttl_seconds: int,
        max_entries: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock if clock is not None else time.monotonic
        self._entries: OrderedDict[str, tuple[float, EvidenceSnapshot]] = OrderedDict()
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def put(self, snapshot: EvidenceSnapshot) -> None:
        """Store one evidence snapshot with a fixed expiration timestamp."""
        now = self._clock()
        expires_at = now + self._ttl_seconds

        # If key already in tombstones, remove it
        self._tombstones.pop(snapshot.snapshot_id, None)

        if snapshot.snapshot_id in self._entries:
            self._entries.pop(snapshot.snapshot_id)
        elif len(self._entries) >= self._max_entries:
            # Capacity eviction (oldest accessed item)
            self._entries.popitem(last=False)

        self._entries[snapshot.snapshot_id] = (expires_at, snapshot)

    def get(self, snapshot_id: str) -> EvidenceSnapshot:
        """Retrieve one snapshot, updating LRU without extending expiration."""
        now = self._clock()

        if snapshot_id in self._entries:
            expires_at, snapshot = self._entries[snapshot_id]
            if now >= expires_at:
                self._entries.pop(snapshot_id)
                self._add_tombstone(snapshot_id)
                raise EventTrailError(
                    "SNAPSHOT_EXPIRED",
                    f"Evidence snapshot {snapshot_id} has expired",
                )
            self._entries.move_to_end(snapshot_id)
            return snapshot

        if snapshot_id in self._tombstones:
            self._tombstones.move_to_end(snapshot_id)
            raise EventTrailError(
                "SNAPSHOT_EXPIRED",
                f"Evidence snapshot {snapshot_id} has expired",
            )

        raise EventTrailError(
            "SNAPSHOT_NOT_FOUND",
            f"Evidence snapshot {snapshot_id} not found",
        )

    def remove(self, snapshot_id: str) -> None:
        """Explicitly remove one snapshot without tombstoning."""
        self._entries.pop(snapshot_id, None)
        self._tombstones.pop(snapshot_id, None)

    def _add_tombstone(self, snapshot_id: str) -> None:
        if len(self._tombstones) >= self._max_entries:
            self._tombstones.popitem(last=False)
        self._tombstones[snapshot_id] = None
