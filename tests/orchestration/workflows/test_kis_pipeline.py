"""Tests for KISPipeline driven by resolved KISIntent."""

import unittest
from unittest.mock import MagicMock, Mock

from hcmai.api.contracts.search import SearchResult, SearchResultMetadata
from hcmai.kis.models import KISEntity, KISEntityBinding, KISEvent, KISIntent, KISTemporalEdge
from hcmai.orchestration.errors import InvalidQueryInputError
from hcmai.orchestration.workflows.kis import KISPipeline, KISSearchExecution


class KISPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = KISIntent(
            revision=2,
            inputs=["A woman talks to a man.", "She takes a white plate."],
            language="en",
            query_text="A woman talks to a man before taking a white plate.",
            entities=[
                KISEntity(id="X1", kind="person", description="woman"),
                KISEntity(id="X2", kind="person", description="man"),
                KISEntity(id="X3", kind="object", description="white plate"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A woman talks to a man",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X2", role="listener"),
                    ],
                ),
                KISEvent(
                    id="E2",
                    text="The woman takes a white plate",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="actor"),
                        KISEntityBinding(entity_id="X3", role="object"),
                    ],
                ),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", relation="before", target="E2"),
            ],
        )

    def test_execute_passes_canonical_intent_events_to_temporal_search(self) -> None:
        corpus = Mock()
        temporal = Mock()
        search_result = Mock()
        search_result.paths = [Mock()]
        search_result.retrieval_ms = 12.0
        search_result.alignment_ms = 3.5
        temporal.search.return_value = search_result

        pipeline = KISPipeline(corpus=corpus, temporal=temporal)
        mock_materializer = Mock()
        mock_result = SearchResult(
            frame_id="v1_f1",
            video_id="v1",
            frame_idx=100,
            timestamp_ms=4000,
            score=0.95,
            frame_ids=["v1_f1", "v1_f2"],
            timestamps_ms=[4000, 8000],
            metadata=SearchResultMetadata(),
        )
        mock_materializer.build_kis_result.return_value = mock_result
        pipeline.materializer = mock_materializer

        retrieval_events = (
            "A woman talks to a man",
            "The woman takes a white plate",
        )
        execution = pipeline.execute(
            intent=self.intent,
            retrieval_events=retrieval_events,
            use_dense=True,
            use_bm25=True,
            top_k=10,
            query_ms=5.0,
        )

        temporal.search.assert_called_once_with(
            ("A woman talks to a man", "The woman takes a white plate"),
            retrieval_events=("A woman talks to a man", "The woman takes a white plate"),
            caption_events=("A woman talks to a man", "The woman takes a white plate"),
            use_dense=True,
            use_bm25=True,
            top_k=10,
        )
        self.assertIsInstance(execution, KISSearchExecution)
        self.assertEqual(execution.results, [mock_result])
        self.assertEqual(execution.latency.query_ms, 5.0)
        self.assertEqual(execution.latency.retrieval_ms, 12.0)
        self.assertEqual(execution.latency.alignment_ms, 3.5)

    def test_execute_disables_bm25_caption_events_when_use_bm25_false(self) -> None:
        corpus = Mock()
        temporal = Mock()
        search_result = Mock()
        search_result.paths = []
        search_result.retrieval_ms = 10.0
        search_result.alignment_ms = 2.0
        temporal.search.return_value = search_result

        pipeline = KISPipeline(corpus=corpus, temporal=temporal)
        pipeline.materializer = Mock()

        retrieval_events = ("A woman talks to a man", "The woman takes a white plate")
        pipeline.execute(
            intent=self.intent,
            retrieval_events=retrieval_events,
            use_dense=True,
            use_bm25=False,
            top_k=5,
        )

        temporal.search.assert_called_once_with(
            ("A woman talks to a man", "The woman takes a white plate"),
            retrieval_events=retrieval_events,
            caption_events=None,
            use_dense=True,
            use_bm25=False,
            top_k=5,
        )

    def test_execute_rejects_cardinality_mismatch(self) -> None:
        corpus = Mock()
        temporal = Mock()
        pipeline = KISPipeline(corpus=corpus, temporal=temporal)

        with self.assertRaises(InvalidQueryInputError):
            pipeline.execute(
                intent=self.intent,
                retrieval_events=["Only one event"],
                use_dense=True,
                use_bm25=False,
                top_k=5,
            )

    def test_execute_requires_loaded_corpus_and_temporal(self) -> None:
        pipeline = KISPipeline(corpus=None, temporal=None)

        with self.assertRaisesRegex(RuntimeError, "canonical frame data is not loaded"):
            pipeline.execute(
                intent=self.intent,
                retrieval_events=["a", "b"],
                use_dense=True,
                use_bm25=False,
                top_k=5,
            )


if __name__ == "__main__":
    unittest.main()
