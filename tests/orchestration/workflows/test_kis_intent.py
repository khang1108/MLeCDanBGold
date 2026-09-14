"""Tests for deterministic KIS intent builder."""

import unittest

from hcmai.orchestration.workflows.kis_intent import KISIntentBuilder


class KISIntentBuilderTest(unittest.TestCase):
    def test_builds_revision_from_ordered_inputs(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "  A woman is standing in a kitchen.  ",
            "She is talking to a man.",
        ])
        self.assertEqual(intent.revision, 2)
        self.assertEqual(
            intent.inputs,
            ["A woman is standing in a kitchen.", "She is talking to a man."],
        )
        self.assertEqual(
            intent.query_text,
            "A woman is standing in a kitchen. She is talking to a man.",
        )
        self.assertEqual(
            intent.events,
            ["A woman is standing in a kitchen", "She is talking to a man"],
        )

    def test_clues_without_terminal_punctuation_still_split_into_distinct_events(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "A woman is standing in a kitchen",
            "She is talking to a man",
        ])
        self.assertEqual(intent.revision, 2)
        self.assertEqual(len(intent.events), 2)
        self.assertEqual(intent.events[0], "A woman is standing in a kitchen")
        self.assertEqual(intent.events[1], "She is talking to a man")

    def test_preserves_input_order(self) -> None:
        intent = KISIntentBuilder(max_temporal_event_count=8).build([
            "First event happens.",
            "Then the second event happens.",
            "Finally the third event happens.",
        ])
        self.assertEqual(intent.revision, 3)
        self.assertEqual(intent.events[0], "First event happens")
        self.assertEqual(intent.events[-1], "Finally the third event happens")

    def test_rejects_intent_that_exceeds_temporal_event_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 2 temporal events"):
            KISIntentBuilder(max_temporal_event_count=2).build([
                "One happens. Two happens. Three happens."
            ])

    def test_rejects_empty_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            KISIntentBuilder().build([])

    def test_rejects_whitespace_only_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            KISIntentBuilder().build(["   "])


if __name__ == "__main__":
    unittest.main()
