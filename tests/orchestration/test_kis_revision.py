"""Tests for SearchService.search_kis_revision orchestration."""

import unittest
from unittest.mock import Mock

from hcmai.api.contracts.kis import KISInput, KISRevisionSearchRequest, KISRevisionSearchResponse
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.kis.models import KISEntity, KISEntityBinding, KISEvent, KISIntent, KISTemporalEdge
from hcmai.orchestration.utils.errors import RevisionConflictError
from hcmai.orchestration.pipeline import SearchService, SearchServiceUnavailableError
from hcmai.orchestration.workflows.kis import KISSearchExecution


class KISRevisionOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_intent_en = KISIntent(
            revision=2,
            language="en",
            query_text="A man enters a room and talks to a woman.",
            entities=[
                KISEntity(id="X1", kind="person", description="man"),
                KISEntity(id="X2", kind="person", description="woman"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A man enters a room",
                    bindings=[KISEntityBinding(entity_id="X1", role="actor")],
                ),
                KISEvent(
                    id="E2",
                    text="The man talks to a woman",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X2", role="listener"),
                    ],
                ),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", relation="before", target="E2"),
            ],
        )

    def test_search_kis_revision_success_english_without_translation(self) -> None:
        corpus = Mock()
        retrieval = Mock()
        intent_resolver = Mock()
        intent_resolver.resolve.return_value = self.mock_intent_en
        event_translator = Mock()

        service = SearchService(
            corpus=corpus,
            retrieval=retrieval,
            temporal_evidence=Mock(),
            intent_resolver=intent_resolver,
            event_translator=event_translator,
        )

        mock_execution = KISSearchExecution(
            results=[
                SearchResult(
                    frame_id="v1_f1",
                    video_id="v1",
                    frame_idx=10,
                    timestamp_ms=1000,
                    score=0.9,
                    frame_ids=["v1_f1", "v1_f2"],
                    timestamps_ms=[1000, 2000],
                    metadata=SearchResultMetadata(),
                )
            ],
            latency=SearchLatency(query_ms=5.0, retrieval_ms=10.0),
        )
        service.kis = Mock()
        service.kis.execute.return_value = mock_execution

        request = KISRevisionSearchRequest(
            inputs=[
                KISInput(text="A man enters a room."),
                KISInput(text="He talks to a woman."),
            ],
            expected_revision=1,
            use_dense=True,
            use_bm25=True,
            top_k=10,
        )

        response = service.search_kis_revision(request)

        intent_resolver.resolve.assert_called_once_with(
            ["A man enters a room.", "He talks to a woman."], revision=2
        )
        event_translator.translate.assert_not_called()
        self.assertEqual(service.kis.execute.call_count, 1)
        call_kwargs = service.kis.execute.call_args.kwargs
        self.assertEqual(call_kwargs["intent"], self.mock_intent_en)
        self.assertEqual(
            call_kwargs["retrieval_plan"].dense_texts,
            ("A man enters a room", "The man talks to a woman"),
        )
        self.assertTrue(call_kwargs["use_dense"])
        self.assertTrue(call_kwargs["use_bm25"])
        self.assertEqual(call_kwargs["top_k"], 10)
        self.assertGreaterEqual(call_kwargs["intent_ms"], 0.0)
        self.assertIsInstance(response, KISRevisionSearchResponse)
        self.assertEqual(response.intent, self.mock_intent_en)
        self.assertEqual(
            [event.dense_text for event in response.exploration_seed.events],
            ["A man enters a room", "The man talks to a woman"],
        )
        self.assertEqual(
            [event.bm25_text for event in response.exploration_seed.events],
            ["A man enters a room", "The man talks to a woman"],
        )
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.exploration_seed.semantic_revision, 2)
        self.assertNotIn("dense_events", response.model_dump())
        self.assertNotIn("bm25_events", response.model_dump())
        self.assertGreaterEqual(call_kwargs["translation_ms"], 0)

    def test_search_kis_revision_translates_vietnamese_for_dense_retrieval(self) -> None:
        intent_vi = KISIntent(
            revision=1,
            language="vi",
            query_text="Một người phụ nữ trong bếp.",
            entities=[KISEntity(id="X1", kind="person", description="phụ nữ")],
            events=[
                KISEvent(
                    id="E1",
                    text="Một người phụ nữ trong bếp",
                    bindings=[KISEntityBinding(entity_id="X1", role="actor")],
                )
            ],
            temporal_edges=[],
        )

        corpus = Mock()
        retrieval = Mock()
        intent_resolver = Mock()
        intent_resolver.resolve.return_value = intent_vi
        event_translator = Mock()
        event_translator.translate.return_value = ("A woman in a kitchen",)

        service = SearchService(
            corpus=corpus,
            retrieval=retrieval,
            temporal_evidence=Mock(),
            intent_resolver=intent_resolver,
            event_translator=event_translator,
        )

        mock_execution = KISSearchExecution(
            results=[],
            latency=SearchLatency(query_ms=2.0),
        )
        service.kis = Mock()
        service.kis.execute.return_value = mock_execution

        request = KISRevisionSearchRequest(
            inputs=[KISInput(text="Một người phụ nữ trong bếp.")],
            expected_revision=0,
            use_dense=True,
            use_bm25=True,
            top_k=5,
        )

        response = service.search_kis_revision(request)

        event_translator.translate.assert_called_once_with(
            ("Một người phụ nữ trong bếp",),
            language="vi",
        )
        self.assertEqual([event.dense_text for event in response.exploration_seed.events], ["A woman in a kitchen"])
        self.assertEqual([event.bm25_text for event in response.exploration_seed.events], ["Một người phụ nữ trong bếp"])

    def test_search_kis_revision_rejects_revision_conflict_before_inference(self) -> None:
        intent_resolver = Mock()
        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            intent_resolver=intent_resolver,
        )

        # inputs has 2 items -> expected_revision should be 1, but client sends 0
        request = KISRevisionSearchRequest(
            inputs=[
                KISInput(text="Clue 1"),
                KISInput(text="Clue 2"),
            ],
            expected_revision=0,
        )

        with self.assertRaises(RevisionConflictError):
            service.search_kis_revision(request)

        intent_resolver.resolve.assert_not_called()

    def test_search_kis_revision_requires_intent_resolver(self) -> None:
        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            intent_resolver=None,
        )
        request = KISRevisionSearchRequest(
            inputs=[KISInput(text="Clue 1")],
            expected_revision=0,
        )

        with self.assertRaises(SearchServiceUnavailableError):
            service.search_kis_revision(request)


if __name__ == "__main__":
    unittest.main()
