"""Tests for KISPipeline driven by resolved KISIntent."""

import unittest
from unittest.mock import Mock

import pytest
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan

from hcmai.api.contracts.search import SearchResult, SearchResultMetadata
from hcmai.kis.models import KISEntity, KISEntityBinding, KISEvent, KISIntent, KISTemporalEdge
from hcmai.orchestration.utils.errors import InvalidQueryInputError
from hcmai.orchestration.workflows.kis import KISPipeline, KISSearchExecution


class KISPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.intent = KISIntent(
            revision=2,
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
        temporal.search_plan_artifact.return_value = Mock(
            result=search_result, video_scores=(), decoder_config=Mock()
        )

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
        plan = make_plan(retrieval_events)
        execution = pipeline.execute(
            intent=self.intent,
            retrieval_plan=plan,
            use_dense=True,
            use_bm25=True,
            top_k=10,
            intent_ms=2.0,
            translation_ms=3.0,
        )

        temporal.search_plan_artifact.assert_called_once_with(
            plan,
            image_component=None,
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
        temporal.search_plan_artifact.return_value = Mock(
            result=search_result, video_scores=(), decoder_config=Mock()
        )

        pipeline = KISPipeline(corpus=corpus, temporal=temporal)
        pipeline.materializer = Mock()

        retrieval_events = ("A woman talks to a man", "The woman takes a white plate")
        plan = make_plan(retrieval_events)
        pipeline.execute(
            intent=self.intent,
            retrieval_plan=plan,
            use_dense=True,
            use_bm25=False,
            top_k=5,
        )

        temporal.search_plan_artifact.assert_called_once_with(
            plan,
            image_component=None,
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
                retrieval_plan=make_plan(["Only one event"]),
                use_dense=True,
                use_bm25=False,
                top_k=5,
            )

    def test_execute_requires_loaded_corpus_and_temporal(self) -> None:
        pipeline = KISPipeline(corpus=None, temporal=None)

        with self.assertRaisesRegex(RuntimeError, "canonical frame data is not loaded"):
            pipeline.execute(
                intent=self.intent,
                retrieval_plan=make_plan(["a", "b"]),
                use_dense=True,
                use_bm25=False,
                top_k=5,
            )


if __name__ == "__main__":
    unittest.main()


def make_plan(texts):
    return KISRetrievalPlan(events=tuple(
        KISRetrievalEvent(f"E{i}", text, text, text)
        for i, text in enumerate(texts, 1)
    ))


def test_task2_step1_plan_preserves_aligned_views_and_is_immutable():
    from dataclasses import FrozenInstanceError
    event = KISRetrievalEvent("E1", "một phụ nữ vào bếp", "a woman enters a kitchen", "một phụ nữ vào bếp")
    plan = KISRetrievalPlan(events=(event,))
    assert plan.event_ids == ("E1",)
    assert plan.dense_texts == ("a woman enters a kitchen",)
    assert plan.bm25_texts == ("một phụ nữ vào bếp",)
    with pytest.raises(FrozenInstanceError):
        event.event_id = "E2"


@pytest.mark.parametrize("ids", [(), ("E2",), ("E1", "E3"), ("E1", "E1")])
def test_task2_step1_plan_rejects_nonsequential_ids(ids):
    with pytest.raises(ValueError):
        KISRetrievalPlan(events=tuple(KISRetrievalEvent(i, "text", "text", None) for i in ids))


def test_task2_step5_pipeline_uses_plan_text_and_accounts_for_translation():
    intent = KISPipelineTest()
    intent.setUp()
    temporal = Mock()
    temporal.search_plan_artifact.return_value = Mock(
        result=Mock(paths=[], retrieval_ms=4, alignment_ms=5),
        video_scores=(),
        decoder_config=Mock(),
    )
    pipeline = KISPipeline(Mock(), temporal)
    plan = KISRetrievalPlan(events=tuple(
        KISRetrievalEvent(f"E{i}", f"canonical {i}", f"dense {i}", f"literal {i}")
        for i in (1, 2)
    ))
    result = pipeline.execute(intent=intent.intent, retrieval_plan=plan, use_dense=True, use_bm25=True, top_k=3, intent_ms=20, translation_ms=30)
    temporal.search_plan_artifact.assert_called_once_with(plan, image_component=None, use_dense=True, use_bm25=True, top_k=3)
    assert result.latency.intent_ms == 20
    assert result.latency.translation_ms == 30
    assert result.latency.query_ms == 50
    assert result.latency.total_ms >= 50


@pytest.mark.parametrize("dense,bm25", [(None, "literal"), ("dense", None)])
def test_task2_step5_pipeline_rejects_missing_enabled_text_rows(dense, bm25):
    intent = KISPipelineTest()
    intent.setUp()
    temporal = Mock()
    plan = KISRetrievalPlan(events=tuple(KISRetrievalEvent(f"E{i}", "canonical", dense, bm25) for i in (1, 2)))
    with pytest.raises(InvalidQueryInputError):
        KISPipeline(Mock(), temporal).execute(intent=intent.intent, retrieval_plan=plan, use_dense=True, use_bm25=True, top_k=3)
    temporal.search.assert_not_called()


def test_kis_pipeline_multimodal_plan_execution():
    import numpy as np
    from hcmai.kis.models import KISEvent, KISImageRef, KISIntent, KISTemporalEdge
    from hcmai.retrieval.evidence.components import TemporalScoreComponent
    
    intent = KISIntent(
        revision=1,
        language="en",
        query_text="E1 and E3",
        events=[
            KISEvent(id="E1", text="text one"),
            KISEvent(id="E2", images=[KISImageRef(asset_id="sha256:img2", content_type="image/png")]),
            KISEvent(id="E3", text="text three", images=[KISImageRef(asset_id="sha256:img3", content_type="image/png")]),
        ],
        temporal_edges=[
            KISTemporalEdge(source="E1", relation="before", target="E2"),
            KISTemporalEdge(source="E2", relation="before", target="E3"),
        ],
    )
    plan = KISRetrievalPlan(events=(
        KISRetrievalEvent("E1", "text one", "text one", "text one"),
        KISRetrievalEvent("E2", None, None, None, image_refs=(KISImageRef(asset_id="sha256:img2", content_type="image/png"),)),
        KISRetrievalEvent("E3", "text three", "text three", "text three", image_refs=(KISImageRef(asset_id="sha256:img3", content_type="image/png"),)),
    ))
    
    image_scorer = Mock()
    image_component = TemporalScoreComponent(
        name="visual_image",
        raw_scores=np.zeros((3, 10), dtype=np.float32),
    )
    image_scorer.score_events.return_value = image_component
    
    temporal = Mock()
    mock_search_result = Mock(paths=[], retrieval_ms=10.0, alignment_ms=5.0)
    temporal.search_plan_artifact.return_value = Mock(
        result=mock_search_result,
        video_scores=(),
        decoder_config=Mock(),
    )
    
    corpus = Mock()
    pipeline = KISPipeline(corpus=corpus, temporal=temporal, image_scorer=image_scorer)
    
    res = pipeline.execute(
        intent=intent,
        retrieval_plan=plan,
        use_dense=True,
        use_bm25=True,
        top_k=5,
    )
    image_scorer.score_events.assert_called_once_with(plan.image_ref_rows)
    temporal.search_plan_artifact.assert_called_once_with(
        plan,
        image_component=image_component,
        use_dense=True,
        use_bm25=True,
        top_k=5,
    )


class FakeTemporal:
    def __init__(self, result):
        self.result = result
        self.search_plan_artifact_calls = 0

    def search_plan_artifact(self, *args, **kwargs):
        self.search_plan_artifact_calls += 1
        return Mock(result=self.result, video_scores=(), decoder_config=Mock())


def test_kis_pipeline_uses_search_plan_artifact_only():
    fake_temporal_result = Mock(paths=[], retrieval_ms=1.0, alignment_ms=1.0)
    corpus = Mock()
    temporal = FakeTemporal(fake_temporal_result)
    pipeline = KISPipeline(corpus, temporal)  # type: ignore[arg-type]
    plan = KISRetrievalPlan(events=(
        KISRetrievalEvent("E1", "canonical 1", "dense 1", "literal 1"),
    ))
    intent = KISIntent(
        revision=1,
        language="en",
        query_text="canonical 1",
        events=[KISEvent(id="E1", text="canonical 1")],
    )
    execution = pipeline.execute(
        intent=intent,
        retrieval_plan=plan,
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    assert temporal.search_plan_artifact_calls == 1
    assert execution.temporal_artifact is not None


def test_no_unittest_mock_imported_in_production_workflows():
    import inspect
    from hcmai.orchestration.workflows import kis as kis_wf
    from hcmai.orchestration.workflows import temporal_exploration as exp_wf
    assert "unittest.mock" not in inspect.getsource(kis_wf)
    assert "unittest.mock" not in inspect.getsource(exp_wf)


