"""Tests for KISIntentResolver with fake LLM client."""

import unittest
from unittest.mock import Mock

from hcmai.kis.models import KISIntent
from hcmai.kis.resolver import KISIntentResolver


class ResolverTest(unittest.TestCase):
    def test_resolver_accepts_reordered_temporal_graph(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = KISIntent(
            revision=2,
            inputs=["A man enters a room.", "Before that, he talks to a woman."],
            language="en",
            query_text="A man talks to a woman before entering a room.",
            entities=[
                {"id": "X1", "kind": "person", "description": "man"},
                {"id": "X2", "kind": "person", "description": "woman"},
                {"id": "X3", "kind": "place", "description": "room"},
            ],
            events=[
                {
                    "id": "E1",
                    "text": "A man talks to a woman",
                    "bindings": [
                        {"entity_id": "X1", "role": "speaker"},
                        {"entity_id": "X2", "role": "conversation partner"},
                    ],
                },
                {
                    "id": "E2",
                    "text": "The man enters a room",
                    "bindings": [
                        {"entity_id": "X1", "role": "person entering"},
                        {"entity_id": "X3", "role": "destination"},
                    ],
                },
            ],
            temporal_edges=[{"source": "E1", "relation": "before", "target": "E2"}],
        )
        resolver = KISIntentResolver(llm)
        intent = resolver.resolve([
            "A man enters a room.",
            "Before that, he talks to a woman.",
        ])
        self.assertEqual([event.id for event in intent.events], ["E1", "E2"])
        llm.generate_structured.assert_called_once()
        messages, response_model = llm.generate_structured.call_args.args
        self.assertIs(response_model, KISIntent)
        self.assertIn("Preserve every supported fact", messages[0]["content"])
        self.assertIn("unless a later clue explicitly corrects it", messages[0]["content"])
        self.assertIn("Clue 1: A man enters a room.", messages[1]["content"])
        self.assertIn("Clue 2: Before that, he talks to a woman.", messages[1]["content"])

    def test_provider_failure_is_not_hidden_by_deterministic_fallback(self) -> None:
        llm = Mock()
        llm.generate_structured.side_effect = RuntimeError("provider down")
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            KISIntentResolver(llm).resolve(["A woman is in a kitchen."])

    def test_rejects_empty_or_whitespace_inputs(self) -> None:
        llm = Mock()
        resolver = KISIntentResolver(llm)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            resolver.resolve([])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            resolver.resolve(["   "])

    def test_rejects_if_llm_altered_revision_or_inputs(self) -> None:
        llm = Mock()
        llm.generate_structured.return_value = KISIntent(
            revision=99,  # altered
            inputs=["Altered clue."],
            language="en",
            query_text="Altered clue.",
            entities=[],
            events=[{"id": "E1", "text": "Altered clue", "bindings": []}],
            temporal_edges=[],
        )
        resolver = KISIntentResolver(llm)
        with self.assertRaisesRegex(ValueError, "changed KIS revision"):
            resolver.resolve(["Original clue."])


if __name__ == "__main__":
    unittest.main()
