"""Tests for SearchService.search_kis stateless semantic orchestration."""

import unittest
from unittest.mock import Mock

from hcmai.api.contracts.kis import (
    EventPatch,
    GlobalRewriteOperation,
    InitialResolveOperation,
    KISSearchRequest,
    KISSearchResponse,
    PatchEventsOperation,
    SearchOnlyOperation,
)
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.kis.assets import KISImageAssetStore
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.scoped_resolver import ScopedResolutionBatch, ScopedResolvedEvent
from hcmai.orchestration.utils.errors import InvalidQueryInputError, RevisionConflictError
from hcmai.orchestration.pipeline import SearchService, SearchServiceUnavailableError
from hcmai.orchestration.workflows.kis import KISSearchExecution


class KISOrchestrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_intent_en = KISIntent(
            revision=1,
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

        self.mock_execution = KISSearchExecution(
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

    def _make_service(
        self,
        *,
        intent_resolver=None,
        scoped_resolver=None,
        global_rewriter=None,
        kis_image_assets=None,
    ) -> SearchService:
        corpus = Mock()
        retrieval = Mock()
        if kis_image_assets is None:
            kis_image_assets = Mock()
            kis_image_assets.ref.side_effect = lambda aid: KISImageRef(
                asset_id=aid, content_type="image/png"
            )
        service = SearchService(
            corpus=corpus,
            retrieval=retrieval,
            temporal_evidence=Mock(),
            intent_resolver=intent_resolver,
            scoped_resolver=scoped_resolver,
            global_rewriter=global_rewriter,
            kis_image_assets=kis_image_assets,
        )
        service.kis = Mock()
        service.kis.execute.return_value = self.mock_execution
        return service

    def test_search_kis_initial_resolve_natural_text_english(self) -> None:
        intent_resolver = Mock()
        intent_resolver.resolve_initial.return_value = self.mock_intent_en

        service = self._make_service(
            intent_resolver=intent_resolver,
        )

        request = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve",
                text="A man enters a room and talks to a woman.",
            ),
            use_dense=True,
            use_bm25=True,
            top_k=10,
        )

        response = service.search_kis(request)

        intent_resolver.resolve_initial.assert_called_once_with(
            "A man enters a room and talks to a woman.", revision=1
        )
        self.assertEqual(response.intent, self.mock_intent_en)
        self.assertEqual(response.operation_summary.kind, "initial_resolve")
        self.assertEqual(response.operation_summary.affected_event_ids, ["E1", "E2"])
        self.assertEqual(len(response.results), 1)

    def test_REQ_004_vietnamese_text_uses_direct_multilingual_views(self) -> None:
        intent = KISIntent(
            revision=1,
            query_text="Một người phụ nữ trong bếp.",
            entities=[],
            events=[KISEvent(id="E1", text="Một người phụ nữ trong bếp.")],
            temporal_edges=[],
        )
        intent_resolver = Mock()
        intent_resolver.resolve_initial.return_value = intent
        service = self._make_service(intent_resolver=intent_resolver)

        request = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve",
                text="Một người phụ nữ trong bếp.",
            ),
            use_dense=True,
            use_bm25=True,
            top_k=5,
        )

        response = service.search_kis(request)

        event = response.exploration_seed.events[0]
        self.assertEqual(event.dense_text, "Một người phụ nữ trong bếp.")
        self.assertEqual(event.bm25_text, "Một người phụ nữ trong bếp.")
        self.assertEqual(response.latency.translation_ms, 0.0)

    def test_search_kis_initial_resolve_image_only_makes_no_llm_call(self) -> None:
        intent_resolver = Mock()
        scoped_resolver = Mock()
        service = self._make_service(
            intent_resolver=intent_resolver,
            scoped_resolver=scoped_resolver,
        )

        request = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve",
                image_refs=[KISImageRef(asset_id="sha256:abc", content_type="image/png")],
            ),
            use_dense=True,
            use_bm25=False,
        )

        response = service.search_kis(request)

        intent_resolver.resolve_initial.assert_not_called()
        scoped_resolver.resolve.assert_not_called()
        self.assertEqual(response.intent.revision, 1)
        self.assertIsNone(response.intent.query_text)
        self.assertEqual(len(response.intent.events), 1)
        self.assertEqual(response.intent.events[0].id, "E1")
        self.assertIsNone(response.intent.events[0].text)
        self.assertEqual(response.intent.events[0].images[0].asset_id, "sha256:abc")
        self.assertEqual(response.operation_summary.kind, "initial_resolve")
        self.assertEqual(response.operation_summary.affected_event_ids, ["E1"])

    def test_search_kis_initial_resolve_text_plus_image(self) -> None:
        scoped_resolver = Mock()
        scoped_resolver.resolve.return_value = ScopedResolutionBatch(
            events=[ScopedResolvedEvent(event_id="E1", text="Woman in kitchen")],
        )

        service = self._make_service(scoped_resolver=scoped_resolver)

        request = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve",
                text="Woman in kitchen",
                image_refs=[KISImageRef(asset_id="sha256:abc", content_type="image/png")],
            ),
            use_dense=True,
            use_bm25=True,
        )

        response = service.search_kis(request)

        self.assertEqual(scoped_resolver.resolve.call_count, 1)
        self.assertEqual(response.intent.revision, 1)
        self.assertEqual(response.intent.events[0].text, "Woman in kitchen")
        self.assertEqual(response.intent.events[0].images[0].asset_id, "sha256:abc")

    def test_search_kis_patch_events_text_instruction(self) -> None:
        scoped_resolver = Mock()
        scoped_resolver.resolve.return_value = ScopedResolutionBatch(
            events=[
                ScopedResolvedEvent(
                    event_id="E2",
                    text="The man is a chef wearing black",
                    bindings=[KISEntityBinding(entity_id="X1", role="actor")],
                )
            ],
        )

        service = self._make_service(scoped_resolver=scoped_resolver)

        request = KISSearchRequest(
            base_intent=self.mock_intent_en,
            expected_revision=1,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[
                    EventPatch(
                        event_id="E2",
                        instruction="the man is a chef wearing black",
                    )
                ],
            ),
            use_dense=True,
            use_bm25=True,
        )

        response = service.search_kis(request)

        self.assertEqual(response.intent.revision, 2)
        self.assertEqual(response.intent.events[0].text, "A man enters a room")
        self.assertEqual(
            response.intent.events[1].text, "The man is a chef wearing black"
        )
        self.assertEqual(response.operation_summary.kind, "patch_events")
        self.assertEqual(response.operation_summary.affected_event_ids, ["E2"])

    def test_search_kis_patch_events_image_only_makes_no_llm_call(self) -> None:
        asset_store = Mock()
        asset_store.ref.return_value = KISImageRef(
            asset_id="sha256:photo", content_type="image/jpeg"
        )
        scoped_resolver = Mock()

        service = self._make_service(
            scoped_resolver=scoped_resolver,
            kis_image_assets=asset_store,
        )

        request = KISSearchRequest(
            base_intent=self.mock_intent_en,
            expected_revision=1,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[
                    EventPatch(
                        event_id="E1",
                        add_image_ids=["sha256:photo"],
                    )
                ],
            ),
            use_dense=True,
            use_bm25=True,
        )

        response = service.search_kis(request)

        scoped_resolver.resolve.assert_not_called()
        self.assertEqual(response.intent.revision, 2)
        self.assertEqual(len(response.intent.events[0].images), 1)
        self.assertEqual(response.intent.events[0].images[0].asset_id, "sha256:photo")
        self.assertEqual(response.operation_summary.kind, "patch_events")

    def test_search_kis_global_rewrite(self) -> None:
        global_rewriter = Mock()
        rewritten_intent = self.mock_intent_en.model_copy(
            update={
                "revision": 2,
                "query_text": "Rewritten global query text.",
            }
        )
        global_rewriter.rewrite.return_value = rewritten_intent

        service = self._make_service(global_rewriter=global_rewriter)

        request = KISSearchRequest(
            base_intent=self.mock_intent_en,
            expected_revision=1,
            operation=GlobalRewriteOperation(
                kind="global_rewrite",
                instruction="Resolve all pronouns explicitly.",
            ),
            use_dense=True,
            use_bm25=True,
        )

        response = service.search_kis(request)

        global_rewriter.rewrite.assert_called_once_with(
            base=self.mock_intent_en,
            instruction="Resolve all pronouns explicitly.",
        )
        self.assertEqual(response.intent.revision, 2)
        self.assertEqual(response.operation_summary.kind, "global_rewrite")
        self.assertEqual(response.operation_summary.affected_event_ids, ["E1", "E2"])

    def test_search_kis_search_only_preserves_revision_and_intent(self) -> None:
        service = self._make_service()

        request = KISSearchRequest(
            base_intent=self.mock_intent_en,
            expected_revision=1,
            operation=SearchOnlyOperation(kind="search_only"),
            use_dense=True,
            use_bm25=False,
            top_k=50,
        )

        response = service.search_kis(request)

        self.assertEqual(response.intent, self.mock_intent_en)
        self.assertEqual(response.intent.revision, 1)
        self.assertEqual(response.operation_summary.kind, "search_only")
        self.assertEqual(response.operation_summary.affected_event_ids, [])

    def test_search_kis_rejects_revision_conflict_before_inference(self) -> None:
        service = self._make_service()

        # Expected 0 when base_intent exists -> conflict
        request = KISSearchRequest(
            base_intent=self.mock_intent_en,
            expected_revision=0,
            operation=SearchOnlyOperation(kind="search_only"),
        )
        with self.assertRaises(RevisionConflictError):
            service.search_kis(request)

        # Expected != 0 when base_intent is None -> conflict
        request_initial_bad_rev = KISSearchRequest(
            base_intent=None,
            expected_revision=2,
            operation=InitialResolveOperation(
                kind="initial_resolve", text="query"
            ),
        )
        with self.assertRaises(RevisionConflictError):
            service.search_kis(request_initial_bad_rev)

        # Non-initial_resolve when base_intent is None -> conflict
        request_patch_without_base = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=SearchOnlyOperation(kind="search_only"),
        )
        with self.assertRaises(RevisionConflictError):
            service.search_kis(request_patch_without_base)

    def test_search_kis_requires_intent_resolver_for_initial_resolve(self) -> None:
        service = self._make_service(intent_resolver=None)
        request = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve", text="query"
            ),
        )

        with self.assertRaises(SearchServiceUnavailableError):
            service.search_kis(request)

    def test_search_kis_builds_plan_for_sparse_text_and_image_events(self) -> None:
        """SearchService translates text-only events and preserves image refs."""
        img2 = KISImageRef(asset_id="sha256:img2", content_type="image/png")
        img3 = KISImageRef(asset_id="sha256:img3", content_type="image/png")
        intent_vi = KISIntent(
            revision=1,
            query_text="người phụ nữ và đĩa",
            events=[
                KISEvent(id="E1", text="người phụ nữ"),
                KISEvent(id="E2", images=[img2]),
                KISEvent(id="E3", text="chiếc đĩa trắng", images=[img3]),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", relation="before", target="E2"),
                KISTemporalEdge(source="E2", relation="before", target="E3"),
            ],
        )

        service = self._make_service()
        request = KISSearchRequest(
            base_intent=intent_vi,
            expected_revision=1,
            operation=SearchOnlyOperation(kind="search_only"),
            use_dense=True,
            use_bm25=True,
        )

        response = service.search_kis(request)

        plan = service.kis.execute.call_args.kwargs["retrieval_plan"]
        self.assertEqual(plan.event_ids, ("E1", "E2", "E3"))

        self.assertEqual(plan.events[0].canonical_text, "người phụ nữ")
        self.assertEqual(plan.events[0].dense_text, "người phụ nữ")
        self.assertEqual(plan.events[0].bm25_text, "người phụ nữ")
        self.assertEqual(plan.events[0].image_refs, ())

        self.assertIsNone(plan.events[1].canonical_text)
        self.assertIsNone(plan.events[1].dense_text)
        self.assertIsNone(plan.events[1].bm25_text)
        self.assertEqual(plan.events[1].image_refs, (img2,))

        self.assertEqual(plan.events[2].canonical_text, "chiếc đĩa trắng")
        self.assertEqual(plan.events[2].dense_text, "chiếc đĩa trắng")
        self.assertEqual(plan.events[2].bm25_text, "chiếc đĩa trắng")
        self.assertEqual(plan.events[2].image_refs, (img3,))

        self.assertEqual(response.exploration_seed.events[0].image_refs, [])
        self.assertEqual(response.exploration_seed.events[1].image_refs, [img2])
        self.assertEqual(response.exploration_seed.events[2].image_refs, [img3])


if __name__ == "__main__":
    unittest.main()
