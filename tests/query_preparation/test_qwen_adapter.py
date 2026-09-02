"""Tests for the thin HCMAI-to-Thundercompute query adapter."""

import pytest
from hcmai.query_preparation.adapters.qwen import QwenQueryPreparationAdapter
from hcmai.query_preparation.service import QueryPreparationError


class FakeGateway:
    """Expose the two structured Thundercompute operations."""

    @staticmethod
    def translate_query_events(events: list[str]) -> list[str]:
        return [f"literal {event}" for event in events]

    @staticmethod
    def generate_query_candidates(events: list[str], candidate_count: int = 5) -> dict[str, object]:
        return {
            "literal_en": [f"literal {event}" for event in events],
            "candidates": [list(events) for _ in range(candidate_count)],
        }



def test_adapter_converts_structured_arrays_to_tuples() -> None:
    """Keep provider conversion thin and deterministic."""

    adapter = QwenQueryPreparationAdapter(FakeGateway())

    literal, candidates = adapter.generate_candidates(("E1", "E2"), 5)

    assert adapter.translate(("E1",)) == ("literal E1",)
    assert literal == ("literal E1", "literal E2")
    assert candidates == (("E1", "E2"),) * 5


def test_adapter_rejects_missing_structured_fields() -> None:
    """Do not parse or guess malformed provider output."""

    class InvalidGateway(FakeGateway):
        def generate_query_candidates(
            self, events: list[str], candidate_count: int = 5
        ) -> dict[str, object]:
            return {"literal_en": list(events)}

    with pytest.raises(QueryPreparationError, match="candidates"):
        QwenQueryPreparationAdapter(InvalidGateway()).generate_candidates(("E1",), 5)