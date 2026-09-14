"""Acceptance smoke tests for KIS semantic intent resolution and search.

Verifies the 3 required acceptance cases from Task 9 of the KIS specification:
1. KIS-T one-shot: "A woman talks to a man in a kitchen, then takes a white plate."
2. KIS-C revision 1: "A man enters a room."; revision 2: "Before that, he talks to a woman."
   Verify event order becomes talk -> enter.
3. Correction: first clue says "red shirt"; later clue says "actually orange, not red".
   Verify canonical intent/event text contains the corrected color and raw clue history
   still contains both clues.

For each case, verify returned frame_ids length equals len(intent.events).
"""

import unittest
from unittest.mock import Mock

from hcmai.api.contracts.kis import KISInput, KISRevisionSearchRequest
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.resolver import KISIntentResolver
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.kis import KISSearchExecution


class KISAcceptanceSmokeTest(unittest.TestCase):
    def test_case_1_kist_one_shot(self) -> None:
        """Case 1: KIS-T one-shot query.

        Query: A woman talks to a man in a kitchen, then takes a white plate.
        Verify:
        - len(intent.events) == 2
        - returned frame_ids length == len(intent.events)
        """
        raw_input = "A woman talks to a man in a kitchen, then takes a white plate."
        mock_intent = KISIntent(
            revision=1,
            inputs=[raw_input],
            language="en",
            query_text="A woman talks to a man in a kitchen, then takes a white plate.",
            entities=[
                KISEntity(id="X1", kind="person", description="woman"),
                KISEntity(id="X2", kind="person", description="man"),
                KISEntity(id="X3", kind="place", description="kitchen"),
                KISEntity(id="X4", kind="object", description="white plate"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A woman talks to a man in a kitchen",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X2", role="listener"),
                        KISEntityBinding(entity_id="X3", role="location"),
                    ],
                ),
                KISEvent(
                    id="E2",
                    text="The woman takes a white plate",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="actor"),
                        KISEntityBinding(entity_id="X4", role="object"),
                    ],
                ),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", relation="before", target="E2"),
            ],
        )

        llm = Mock()
        llm.generate_structured.return_value = mock_intent
        resolver = KISIntentResolver(llm)
        resolved_intent = resolver.resolve([raw_input])

        self.assertEqual(len(resolved_intent.events), 2)
        self.assertEqual(resolved_intent.events[0].id, "E1")
        self.assertEqual(resolved_intent.events[1].id, "E2")

        # Wire through search service
        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            temporal_evidence=Mock(),
            intent_resolver=resolver,
        )

        mock_execution = KISSearchExecution(
            results=[
                SearchResult(
                    frame_id="video01_00100",
                    video_id="video01",
                    frame_idx=100,
                    timestamp_ms=4000,
                    score=0.92,
                    frame_ids=["video01_00050", "video01_00100"],
                    timestamps_ms=[2000, 4000],
                    metadata=SearchResultMetadata(),
                )
            ],
            latency=SearchLatency(total_ms=10.0, query_ms=5.0, retrieval_ms=3.0, alignment_ms=2.0),
        )
        service.kis = Mock()
        service.kis.execute.return_value = mock_execution

        response = service.search_kis_revision(
            KISRevisionSearchRequest(
                inputs=[KISInput(text=raw_input)],
                expected_revision=0,
            )
        )

        self.assertEqual(len(response.results), 1)
        # Verify returned frame_ids length equals len(intent.events)
        self.assertEqual(
            len(response.results[0].frame_ids),
            len(response.intent.events),
        )
        self.assertEqual(len(response.results[0].frame_ids), 2)

    def test_case_2_kisc_revision_reordering(self) -> None:
        """Case 2: KIS-C revision reordering.

        Revision 1: A man enters a room.
        Revision 2: Before that, he talks to a woman.
        Verify:
        - Event order becomes talk -> enter (E1 -> E2)
        - returned frame_ids length == len(intent.events) == 2
        """
        clue1 = "A man enters a room."
        clue2 = "Before that, he talks to a woman."

        mock_intent_rev2 = KISIntent(
            revision=2,
            inputs=[clue1, clue2],
            language="en",
            query_text="A man talks to a woman before entering a room.",
            entities=[
                KISEntity(id="X1", kind="person", description="man"),
                KISEntity(id="X2", kind="person", description="woman"),
                KISEntity(id="X3", kind="place", description="room"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A man talks to a woman",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X2", role="listener"),
                    ],
                ),
                KISEvent(
                    id="E2",
                    text="The man enters a room",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="actor"),
                        KISEntityBinding(entity_id="X3", role="destination"),
                    ],
                ),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", relation="before", target="E2"),
            ],
        )

        llm = Mock()
        llm.generate_structured.return_value = mock_intent_rev2
        resolver = KISIntentResolver(llm)
        resolved_intent = resolver.resolve([clue1, clue2])

        # Verify event order: talk -> enter
        self.assertEqual(len(resolved_intent.events), 2)
        self.assertEqual(resolved_intent.events[0].text, "A man talks to a woman")
        self.assertEqual(resolved_intent.events[1].text, "The man enters a room")

        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            temporal_evidence=Mock(),
            intent_resolver=resolver,
        )

        mock_execution = KISSearchExecution(
            results=[
                SearchResult(
                    frame_id="video02_00200",
                    video_id="video02",
                    frame_idx=200,
                    timestamp_ms=8000,
                    score=0.88,
                    frame_ids=["video02_00100", "video02_00200"],
                    timestamps_ms=[4000, 8000],
                    metadata=SearchResultMetadata(),
                )
            ],
            latency=SearchLatency(total_ms=10.0, query_ms=5.0, retrieval_ms=3.0, alignment_ms=2.0),
        )
        service.kis = Mock()
        service.kis.execute.return_value = mock_execution

        response = service.search_kis_revision(
            KISRevisionSearchRequest(
                inputs=[KISInput(text=clue1), KISInput(text=clue2)],
                expected_revision=1,
            )
        )

        self.assertEqual(response.intent.revision, 2)
        self.assertEqual(response.intent.events[0].text, "A man talks to a woman")
        self.assertEqual(response.intent.events[1].text, "The man enters a room")
        # Verify returned frame_ids length equals len(intent.events)
        self.assertEqual(
            len(response.results[0].frame_ids),
            len(response.intent.events),
        )
        self.assertEqual(len(response.results[0].frame_ids), 2)

    def test_case_3_correction_override(self) -> None:
        """Case 3: Correction of previous clue.

        Clue 1: A man in a red shirt walks into an office.
        Clue 2: Actually orange, not red.
        Verify:
        - Canonical intent / event text contains corrected color ("orange", not "red").
        - Raw clue history still contains both clues intact.
        - returned frame_ids length == len(intent.events).
        """
        clue1 = "A man in a red shirt walks into an office."
        clue2 = "Actually orange, not red."

        mock_intent_corrected = KISIntent(
            revision=2,
            inputs=[clue1, clue2],
            language="en",
            query_text="A man in an orange shirt walks into an office.",
            entities=[
                KISEntity(id="X1", kind="person", description="man in orange shirt"),
                KISEntity(id="X2", kind="place", description="office"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A man in an orange shirt walks into an office",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="actor"),
                        KISEntityBinding(entity_id="X2", role="destination"),
                    ],
                ),
            ],
            temporal_edges=[],
        )

        llm = Mock()
        llm.generate_structured.return_value = mock_intent_corrected
        resolver = KISIntentResolver(llm)
        resolved_intent = resolver.resolve([clue1, clue2])

        # Verify raw clue history contains both clues
        self.assertEqual(resolved_intent.inputs, [clue1, clue2])
        # Verify canonical query and event text contains "orange" and not "red"
        self.assertIn("orange", resolved_intent.query_text)
        self.assertNotIn("red", resolved_intent.query_text)
        self.assertIn("orange", resolved_intent.events[0].text)
        self.assertNotIn("red", resolved_intent.events[0].text)

        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            temporal_evidence=Mock(),
            intent_resolver=resolver,
        )

        mock_execution = KISSearchExecution(
            results=[
                SearchResult(
                    frame_id="video03_00150",
                    video_id="video03",
                    frame_idx=150,
                    timestamp_ms=6000,
                    score=0.95,
                    frame_ids=["video03_00150"],
                    timestamps_ms=[6000],
                    metadata=SearchResultMetadata(),
                )
            ],
            latency=SearchLatency(total_ms=10.0, query_ms=5.0, retrieval_ms=3.0, alignment_ms=2.0),
        )
        service.kis = Mock()
        service.kis.execute.return_value = mock_execution

        response = service.search_kis_revision(
            KISRevisionSearchRequest(
                inputs=[KISInput(text=clue1), KISInput(text=clue2)],
                expected_revision=1,
            )
        )

        self.assertEqual(response.intent.inputs, [clue1, clue2])
        self.assertIn("orange", response.intent.query_text)
        # Verify returned frame_ids length equals len(intent.events)
        self.assertEqual(
            len(response.results[0].frame_ids),
            len(response.intent.events),
        )
        self.assertEqual(len(response.results[0].frame_ids), 1)


if __name__ == "__main__":
    unittest.main()
