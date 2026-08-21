from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from hcmai.temporal.state import (
    ProgressiveSearchState,
    ProgressiveStateConflictError,
    ProgressiveStateStore,
    StaleProgressiveStateError,
)


def test_create_commit_failure_and_expiration_are_transactional():
    now = [100.0]
    store = ProgressiveStateStore(10, 2, clock=lambda: now[0])
    proposed = ProgressiveSearchState(search_id="s1", last_snapshot="H1")
    committed = store.create(proposed)
    assert committed.version == 1

    changed = committed.clone()
    changed.last_snapshot = "H1 H2"
    committed2 = store.commit(changed, expected_version=1)
    assert committed2.version == 2
    with pytest.raises(ProgressiveStateConflictError):
        store.commit(changed, expected_version=1)
    assert store.get("s1").last_snapshot == "H1 H2"

    now[0] = 111.0
    assert store.cleanup() == 1
    with pytest.raises(StaleProgressiveStateError):
        store.get("s1")


def test_max_entries_evicts_oldest_state():
    store = ProgressiveStateStore(100, 2)
    store.create(ProgressiveSearchState(search_id="s1"))
    store.create(ProgressiveSearchState(search_id="s2"))
    store.create(ProgressiveSearchState(search_id="s3"))
    assert len(store) == 2
    with pytest.raises(StaleProgressiveStateError):
        store.get("s1")


def test_per_search_lock_serializes_updates():
    store = ProgressiveStateStore(100, 10)
    store.create(ProgressiveSearchState(search_id="s1"))

    def update():
        with store.serialized("s1"):
            current = store.get("s1")
            return store.commit(current, expected_version=current.version).version

    with ThreadPoolExecutor(max_workers=2) as pool:
        versions = sorted(pool.map(lambda _: update(), range(2)))
    assert versions == [2, 3]
