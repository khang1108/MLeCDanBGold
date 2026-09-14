"""Tests for provider-agnostic QueryPreparationService."""

import unittest
from unittest.mock import Mock

from hcmai.common.config import QueryPreparationConfig
from hcmai.query_preparation.models import CandidateBundle, LiteralTranslation
from hcmai.query_preparation.service import (
    QueryPreparationError,
    QueryPreparationService,
)


class QueryPreparationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QueryPreparationConfig(
            model_name="test-model",
            candidate_count=5,
            cache_enabled=True,
        )

    def test_translate_literal_english_is_noop_without_llm_call(self) -> None:
        llm = Mock()
        service = QueryPreparationService(llm, self.config)
        events = ("A woman in a kitchen", "She talks to a man")

        result = service.translate_literal(events, language="en")
        self.assertEqual(result, events)
        llm.generate_structured.assert_not_called()

    def test_translate_literal_vietnamese_calls_llm(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = LiteralTranslation(
            events=["A woman in a kitchen", "A woman talks to a man"]
        )
        service = QueryPreparationService(llm, self.config)
        events = ("Một người phụ nữ trong bếp", "Người phụ nữ nói chuyện với một người đàn ông")

        result = service.translate_literal(events, language="vi")
        self.assertEqual(result, ("A woman in a kitchen", "A woman talks to a man"))
        llm.generate_structured.assert_called_once()

    def test_translate_literal_rejects_cardinality_change(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = LiteralTranslation(
            events=["Only one event"]
        )
        service = QueryPreparationService(llm, self.config)
        events = ("Một người phụ nữ", "Nói chuyện với đàn ông")

        with self.assertRaisesRegex(QueryPreparationError, "changed event count"):
            service.translate_literal(events, language="vi")

    def test_translate_literal_preserves_required_tokens(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = LiteralTranslation(
            events=["A woman holding a cup"]
        )
        service = QueryPreparationService(llm, self.config)
        events = ("Người phụ nữ cầm vật thể OBJ_42",)

        with self.assertRaisesRegex(QueryPreparationError, "omitted required token 'OBJ_42'"):
            service.translate_literal(events, language="vi")

    def test_generate_candidates_produces_aligned_candidate_set(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = CandidateBundle(
            literal_en=["A woman enters", "A woman cooks"],
            candidates=[
                ["Woman walks in", "Woman prepares food"],
                ["A female enters room", "She is cooking meal"],
                ["Lady steps inside", "Lady makes dish"],
                ["Woman comes in", "Woman boiling something"],
                ["A person arrives", "A person is cooking"],
            ],
        )
        service = QueryPreparationService(llm, self.config)
        events = ("Người phụ nữ đi vào", "Người phụ nữ nấu ăn")

        candidate_set = service.generate_candidates(events)
        self.assertEqual(candidate_set.original_events, events)
        self.assertEqual(candidate_set.literal_en, ("A woman enters", "A woman cooks"))
        self.assertEqual(len(candidate_set.candidates), 5)
        for idx, candidate in enumerate(candidate_set.candidates, start=1):
            self.assertEqual(candidate.index, idx)
            self.assertEqual(len(candidate.events), 2)

    def test_generate_candidates_rejects_invalid_bundle_count(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = CandidateBundle(
            literal_en=["A woman enters"],
            candidates=[
                ["Bundle 1"],
                ["Bundle 2"],
            ],
        )
        service = QueryPreparationService(llm, self.config)
        events = ("Người phụ nữ đi vào",)

        with self.assertRaisesRegex(QueryPreparationError, "exactly 5 bundles"):
            service.generate_candidates(events)


if __name__ == "__main__":
    unittest.main()
