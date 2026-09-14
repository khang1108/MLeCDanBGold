"""Tests for KIS semantic intent models and cross-reference validation."""

import unittest

from pydantic import ValidationError

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISTemporalEdge,
)


VALID = {
    "revision": 2,
    "inputs": ["A woman is in a kitchen.", "She talks to a man."],
    "language": "en",
    "query_text": "A woman talks to a man in a kitchen.",
    "entities": [
        {"id": "X1", "kind": "person", "description": "woman in a kitchen"},
        {"id": "X2", "kind": "person", "description": "man talking with X1"},
    ],
    "events": [
        {
            "id": "E1",
            "text": "A woman talks to a man in a kitchen",
            "bindings": [
                {"entity_id": "X1", "role": "speaker"},
                {"entity_id": "X2", "role": "conversation partner"},
            ],
        }
    ],
    "temporal_edges": [],
}


class KISIntentModelTest(unittest.TestCase):
    def test_accepts_resolved_entity_event_graph(self) -> None:
        intent = KISIntent.model_validate(VALID)
        self.assertEqual(intent.events[0].id, "E1")
        self.assertEqual(len(intent.entities), 2)
        self.assertEqual(intent.language, "en")

    def test_rejects_unknown_entity_binding(self) -> None:
        invalid = {
            **VALID,
            "events": [
                {
                    **VALID["events"][0],
                    "bindings": [{"entity_id": "X9", "role": "speaker"}],
                }
            ],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_temporal_edge_against_canonical_order(self) -> None:
        invalid = {
            **VALID,
            "events": [
                {"id": "E1", "text": "first", "bindings": []},
                {"id": "E2", "text": "second", "bindings": []},
            ],
            "temporal_edges": [{"source": "E2", "relation": "before", "target": "E1"}],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_self_temporal_edge(self) -> None:
        invalid = {
            **VALID,
            "events": [
                {"id": "E1", "text": "first", "bindings": []},
            ],
            "temporal_edges": [{"source": "E1", "relation": "before", "target": "E1"}],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_unknown_event_in_temporal_edge(self) -> None:
        invalid = {
            **VALID,
            "events": [
                {"id": "E1", "text": "first", "bindings": []},
            ],
            "temporal_edges": [{"source": "E1", "relation": "before", "target": "E9"}],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_duplicate_entity_ids(self) -> None:
        invalid = {
            **VALID,
            "entities": [
                {"id": "X1", "kind": "person", "description": "one"},
                {"id": "X1", "kind": "person", "description": "two"},
            ],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_non_sequential_event_ids(self) -> None:
        invalid = {
            **VALID,
            "events": [
                {"id": "E1", "text": "first", "bindings": []},
                {"id": "E3", "text": "third", "bindings": []},
            ],
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)

    def test_rejects_exceeding_max_temporal_event_count(self) -> None:
        events = [
            {"id": f"E{i+1}", "text": f"event {i+1}", "bindings": []}
            for i in range(DEFAULT_MAX_TEMPORAL_EVENT_COUNT + 1)
        ]
        invalid = {
            **VALID,
            "events": events,
        }
        with self.assertRaises(ValidationError):
            KISIntent.model_validate(invalid)


if __name__ == "__main__":
    unittest.main()
