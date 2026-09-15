"""Tests for KIS semantic intent models and cross-reference validation."""

import unittest

from pydantic import ValidationError

from hcmai.common.config import DEFAULT_MAX_TEMPORAL_EVENT_COUNT
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)


VALID = {
    "revision": 7,
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
        self.assertEqual(intent.revision, 7)

    def test_accepts_image_only_event(self) -> None:
        event = KISEvent(
            id="E1",
            text=None,
            images=[KISImageRef(asset_id="sha256:abc", content_type="image/png")],
            bindings=[],
        )

        self.assertIsNone(event.text)
        self.assertEqual(event.images[0].asset_id, "sha256:abc")

    def test_rejects_event_without_text_or_images(self) -> None:
        with self.assertRaisesRegex(ValueError, "text or image"):
            KISEvent(id="E1", text=None, images=[], bindings=[])

    def test_accepts_mixed_event_evidence(self) -> None:
        event = KISEvent(
            id="E1",
            text="A woman holds a plate",
            images=[KISImageRef(asset_id="sha256:mixed", content_type="image/webp")],
        )

        self.assertEqual(event.text, "A woman holds a plate")
        self.assertEqual(event.images[0].content_type, "image/webp")

    def test_accepts_image_only_intent_without_language_or_query_text(self) -> None:
        intent = KISIntent(
            revision=11,
            language=None,
            query_text=None,
            entities=[],
            events=[
                KISEvent(
                    id="E1",
                    text=None,
                    images=[
                        KISImageRef(
                            asset_id="sha256:image", content_type="image/jpeg"
                        )
                    ],
                )
            ],
            temporal_edges=[],
        )

        self.assertEqual(intent.revision, 11)

    def test_rejects_missing_language_or_query_for_textual_intent(self) -> None:
        invalid = {**VALID, "language": None, "query_text": None}

        with self.assertRaisesRegex(ValidationError, "image-only"):
            KISIntent.model_validate(invalid)

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
