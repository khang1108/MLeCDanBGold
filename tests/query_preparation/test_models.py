"""Tests for immutable query-preparation domain models."""

from dataclasses import FrozenInstanceError

import pytest
from hcmai.query_preparation.models import QueryCandidate, QueryCandidateSet


def test_query_candidate_is_immutable() -> None:
    """Prevent downstream code from rewriting candidate positions."""

    candidate = QueryCandidate(index=1, events=("E1", "E2"))

    with pytest.raises(FrozenInstanceError):
        candidate.index = 2  # type: ignore[misc]


def test_query_candidate_set_preserves_positional_events() -> None:
    """Represent each event bundle as an ordered tuple."""

    result = QueryCandidateSet(
        original_events=("E1", "E2"),
        literal_en=("L1", "L2"),
        candidates=(QueryCandidate(index=1, events=("C1", "C2")),),
    )

    assert result.original_events == ("E1", "E2")
    assert result.literal_en == ("L1", "L2")
    assert result.candidates[0].events == ("C1", "C2")