"""Tests for the bounded process-local query-preparation cache."""

from hcmai.query_preparation.cache import QueryPreparationCache, cache_key


def test_cache_key_separates_operation_and_preserves_named_token_case() -> None:
    """Keep operations distinct without erasing case-sensitive entities."""

    common = {
        "events": ("  giữ   X  ",),
        "model_name": "qwen",
        "model_revision": "a" * 40,
        "prompt_version": "v1",
    }

    translate = cache_key(operation="translate", **common)
    candidates = cache_key(operation="candidates", **common)
    lowercase = cache_key(operation="translate", **{**common, "events": ("giữ x",)})

    assert translate != candidates
    assert translate != lowercase
    assert "giữ X" in translate


def test_cache_expires_and_evicts_least_recently_used_entry() -> None:
    """Bound memory and refresh recency on successful reads."""

    now = [100.0]
    cache = QueryPreparationCache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    cache.put(("a",), "A")
    cache.put(("b",), "B")
    assert cache.get(("a",)) == "A"

    cache.put(("c",), "C")

    assert cache.get(("b",)) is None
    assert cache.get(("a",)) == "A"
    now[0] = 111.0
    assert cache.get(("a",)) is None
    assert cache.get(("c",)) is None