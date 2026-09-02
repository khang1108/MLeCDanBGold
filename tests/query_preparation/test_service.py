"""Tests for query-preparation orchestration and validation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hcmai.common.config import QueryPreparationConfig
from hcmai.query_preparation.service import QueryPreparationError, QueryPreparationService


class ScriptedAdapter:
    """Return scripted structured outputs and expose call counts."""

    def __init__(self) -> None:
        self.translation: tuple[str, ...] = ("literal X",)
        self.outputs: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = []
        self.translate_calls = 0
        self.generate_calls = 0

    def translate(self, events_vi: Sequence[str]) -> tuple[str, ...]:
        """Return the configured translation."""

        self.translate_calls += 1
        return self.translation

    def generate_candidates(
        self, events_vi: Sequence[str], candidate_count: int
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        """Pop one configured generation result."""

        self.generate_calls += 1
        return self.outputs.pop(0)


def _service(adapter: ScriptedAdapter) -> QueryPreparationService:
    return QueryPreparationService(adapter, QueryPreparationConfig())


def _valid_output(token: str = "X") -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    return (
        (f"literal {token}",),
        tuple((f"candidate {index} {token}",) for index in range(5)),
    )


def test_generate_candidates_retries_once_then_succeeds() -> None:
    """Retry one malformed candidate response and cache only the valid result."""

    adapter = ScriptedAdapter()
    adapter.outputs = [
        (("literal X",), (("only one X",),)),
        _valid_output(),
    ]
    service = _service(adapter)

    result = service.generate_candidates(("mot su kien X",))

    assert len(result.candidates) == 5
    assert adapter.generate_calls == 2


def test_event_count_mismatch_is_rejected() -> None:
    """Never accept translation output that changes positional event count."""

    adapter = ScriptedAdapter()
    adapter.translation = ("one",)

    with pytest.raises(QueryPreparationError, match="event count"):
        _service(adapter).translate_literal(("E1", "E2"))


def test_generation_fails_after_at_most_two_attempts() -> None:
    """Return an explicit error rather than a partial candidate set."""

    adapter = ScriptedAdapter()
    adapter.outputs = [
        (("literal X",), (("one X",),)),
        (("literal X",), (("one X",),)),
    ]

    with pytest.raises(QueryPreparationError, match="exactly 5"):
        _service(adapter).generate_candidates(("event X",))

    assert adapter.generate_calls == 2


def test_valid_results_are_cached_by_normalized_events() -> None:
    """Collapse whitespace for cache identity without a second inference call."""

    adapter = ScriptedAdapter()
    adapter.outputs = [_valid_output()]
    service = _service(adapter)

    first = service.generate_candidates((" event   X ",))
    second = service.generate_candidates(("event X",))

    assert first == second
    assert adapter.generate_calls == 1


def test_placeholder_must_survive_every_output() -> None:
    """Reject candidates that erase an exact unknown-entity placeholder."""

    adapter = ScriptedAdapter()
    invalid = (
        ("literal X 2",),
        tuple((f"candidate {index}",) for index in range(5)),
    )
    adapter.outputs = [invalid, invalid]

    with pytest.raises(QueryPreparationError, match="required token 'X'"):
        _service(adapter).generate_candidates(("cam 2 con X",))