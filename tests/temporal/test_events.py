"""Tests for temporal event normalization."""

import unittest

from hcmai.temporal.events import normalize_event_texts


class TemporalEventsTest(unittest.TestCase):
    def test_normalize_event_texts_strips_internal_whitespace(self) -> None:
        raw = ["  a woman   walks in  ", "she   talks\nto a man\t"]
        normalized = normalize_event_texts(raw)
        self.assertEqual(normalized, ("a woman walks in", "she talks to a man"))

    def test_normalize_event_texts_rejects_string_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "events must be a non-string sequence"):
            normalize_event_texts("not a sequence")  # type: ignore

    def test_normalize_event_texts_rejects_empty_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "events must not be empty"):
            normalize_event_texts([])

    def test_normalize_event_texts_rejects_non_string_elements(self) -> None:
        with self.assertRaisesRegex(ValueError, "events must contain strings"):
            normalize_event_texts(["valid", 123])  # type: ignore

    def test_normalize_event_texts_rejects_empty_or_whitespace_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "events must contain non-empty strings"):
            normalize_event_texts(["valid", "   "])


if __name__ == "__main__":
    unittest.main()
