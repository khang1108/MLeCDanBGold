"""S0/S1 End-to-end acceptance smoke test around progressive multimodal KIS operations."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock

import yaml

from hcmai.api.contracts.kis import (
    EventPatch,
    GlobalRewriteOperation,
    InitialResolveOperation,
    KISSearchRequest,
    PatchEventsOperation,
    SearchOnlyOperation,
)
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.inference.errors import InferenceResponseError
from hcmai.kis.assets import KISImageAssetStore
from hcmai.kis.models import (
    KISEntity,
    KISEntityBinding,
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)
from hcmai.kis.resolution import ScopedResolutionBatch, ScopedResolvedEvent
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.kis import KISSearchExecution


class KISMultimodalAcceptanceSmokeTest(unittest.TestCase):
    """Verifies the complete 8-step semantic operations lifecycle and contracts."""

    def setUp(self) -> None:
        self.mock_execution = KISSearchExecution(
            results=[
                SearchResult(
                    frame_id="V01_00100",
                    video_id="V01",
                    frame_idx=100,
                    timestamp_ms=4000,
                    score=0.92,
                    frame_ids=["V01_00100", "V01_00200"],
                    timestamps_ms=[4000, 8000],
                    metadata=SearchResultMetadata(caption="Kitchen scene"),
                )
            ],
            latency=SearchLatency(query_ms=2.0, retrieval_ms=10.0),
        )

        self.mock_intent_rev1 = KISIntent(
            revision=1,
            query_text="A woman is standing in a kitchen, then talks to a man.",
            entities=[
                KISEntity(id="X1", kind="person", description="woman"),
                KISEntity(id="X2", kind="place", description="kitchen"),
                KISEntity(id="X3", kind="person", description="man"),
            ],
            events=[
                KISEvent(
                    id="E1",
                    text="A woman is standing in a kitchen",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="actor"),
                        KISEntityBinding(entity_id="X2", role="location"),
                    ],
                ),
                KISEvent(
                    id="E2",
                    text="The woman talks to a man",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X3", role="listener"),
                    ],
                ),
            ],
            temporal_edges=[KISTemporalEdge(source="E1", relation="before", target="E2")],
        )

    def _make_service(
        self,
        *,
        intent_resolver=None,
        scoped_resolver=None,
        global_rewriter=None,
        kis_image_assets=None,
    ) -> SearchService:
        service = SearchService(
            corpus=Mock(),
            retrieval=Mock(),
            temporal_evidence=Mock(),
            intent_resolver=intent_resolver,
            scoped_resolver=scoped_resolver,
            global_rewriter=global_rewriter,
            kis_image_assets=kis_image_assets,
        )
        service.kis = Mock()
        service.kis.execute.return_value = self.mock_execution
        return service

    def test_progressive_multimodal_kis_lifecycle(self) -> None:
        """Execute the 8-step semantic operations sequence through SearchService."""
        intent_resolver = Mock()
        scoped_resolver = Mock()
        global_rewriter = Mock()
        image_assets = Mock(spec=KISImageAssetStore)
        image_assets.ref.side_effect = lambda aid: KISImageRef(asset_id=aid, content_type="image/jpeg")

        service = self._make_service(
            intent_resolver=intent_resolver,
            scoped_resolver=scoped_resolver,
            global_rewriter=global_rewriter,
            kis_image_assets=image_assets,
        )

        # ---------------------------------------------------------------------
        # 1. InitialResolve: natural text -> Rev 1
        # ---------------------------------------------------------------------
        intent_resolver.resolve_initial.return_value = self.mock_intent_rev1

        req_1 = KISSearchRequest(
            base_intent=None,
            expected_revision=0,
            operation=InitialResolveOperation(
                kind="initial_resolve",
                text="A woman is standing in a kitchen, then talks to a man.",
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_1 = service.search_kis(req_1)

        self.assertEqual(res_1.intent.revision, 1)
        self.assertEqual([e.id for e in res_1.intent.events], ["E1", "E2"])
        # DP execution check
        call_1 = service.kis.execute.call_args[1]
        self.assertEqual(call_1["retrieval_plan"].event_ids, ("E1", "E2"))
        self.assertEqual(call_1["retrieval_plan"].events[0].canonical_text, "A woman is standing in a kitchen")
        self.assertEqual(call_1["retrieval_plan"].events[0].dense_text, "A woman is standing in a kitchen")
        self.assertEqual(call_1["retrieval_plan"].events[0].bm25_text, "A woman is standing in a kitchen")

        # ---------------------------------------------------------------------
        # 2. PatchEvents: update E2 -> Rev 2
        # ---------------------------------------------------------------------
        scoped_resolver.resolve.return_value = ScopedResolutionBatch(
            events=[
                ScopedResolvedEvent(
                    event_id="E2",
                    text="The woman talks to a chef wearing black",
                    bindings=[
                        KISEntityBinding(entity_id="X1", role="speaker"),
                        KISEntityBinding(entity_id="X3", role="listener"),
                    ],
                )
            ],
        )

        req_2 = KISSearchRequest(
            base_intent=res_1.intent,
            expected_revision=1,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[
                    EventPatch(event_id="E2", instruction="the man is actually a chef wearing black"),
                ],
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_2 = service.search_kis(req_2)

        self.assertEqual(res_2.intent.revision, 2)
        self.assertEqual(res_2.intent.events[1].text, "The woman talks to a chef wearing black")
        self.assertEqual(res_2.intent.events[0].text, self.mock_intent_rev1.events[0].text)

        # ---------------------------------------------------------------------
        # 3. Upload image + PatchEvents: attach image to E2 without text -> Rev 3
        # ---------------------------------------------------------------------
        llm_call_count_before = (
            intent_resolver.resolve_initial.call_count
            + scoped_resolver.resolve.call_count
            + global_rewriter.rewrite.call_count
        )

        req_3 = KISSearchRequest(
            base_intent=res_2.intent,
            expected_revision=2,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[
                    EventPatch(event_id="E2", instruction=None, add_image_ids=["ast_chef_1"]),
                ],
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_3 = service.search_kis(req_3)

        llm_call_count_after = (
            intent_resolver.resolve_initial.call_count
            + scoped_resolver.resolve.call_count
            + global_rewriter.rewrite.call_count
        )
        # Verify LLM was NOT called for image-only attachment
        self.assertEqual(llm_call_count_before, llm_call_count_after)
        self.assertEqual(res_3.intent.revision, 3)
        self.assertEqual(res_3.intent.events[1].images[0].asset_id, "ast_chef_1")
        plan_3 = service.kis.execute.call_args[1]["retrieval_plan"]
        self.assertEqual(plan_3.events[1].image_refs[0].asset_id, "ast_chef_1")

        # ---------------------------------------------------------------------
        # 4. Separate image-only base intent, then patch existing E1 with text
        # ---------------------------------------------------------------------
        img_only_base = KISIntent(
            revision=1,
            query_text=None,
            entities=[],
            events=[
                KISEvent(
                    id="E1",
                    text=None,
                    images=[KISImageRef(asset_id="ast_plate_photo", content_type="image/jpeg")],
                    bindings=[],
                )
            ],
            temporal_edges=[],
        )
        scoped_resolver.resolve.return_value = ScopedResolutionBatch(
            events=[
                ScopedResolvedEvent(
                    event_id="E1",
                    text="the woman is holding a plate",
                    bindings=[],
                )
            ],
        )

        req_4 = KISSearchRequest(
            base_intent=img_only_base,
            expected_revision=1,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[EventPatch(event_id="E1", instruction="the woman is holding a plate")],
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_4 = service.search_kis(req_4)
        self.assertEqual(res_4.intent.events[0].text, "the woman is holding a plate")
        self.assertEqual(res_4.intent.events[0].images[0].asset_id, "ast_plate_photo")

        # ---------------------------------------------------------------------
        # 5. Append on the main flow: E3 -> Rev 4
        # ---------------------------------------------------------------------
        scoped_resolver.resolve.return_value = ScopedResolutionBatch(
            events=[
                ScopedResolvedEvent(
                    event_id="E3",
                    text="then she takes a white plate",
                    bindings=[],
                )
            ],
        )

        req_5 = KISSearchRequest(
            base_intent=res_3.intent,
            expected_revision=3,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[EventPatch(event_id="E3", instruction="then she takes a white plate")],
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_5 = service.search_kis(req_5)
        self.assertEqual(res_5.intent.revision, 4)
        self.assertEqual([e.id for e in res_5.intent.events], ["E1", "E2", "E3"])
        call_5 = service.kis.execute.call_args[1]
        self.assertEqual(call_5["retrieval_plan"].event_ids, ("E1", "E2", "E3"))

        # ---------------------------------------------------------------------
        # 6. GlobalRewrite: resolve all references explicitly -> Rev 5
        # ---------------------------------------------------------------------
        intent_rev5 = KISIntent(
            revision=5,
            query_text="A woman stands in kitchen. She speaks with a chef wearing black. Finally, she takes a white plate.",
            entities=res_5.intent.entities,
            events=[
                KISEvent(id="E1", text="Woman standing in kitchen", bindings=res_5.intent.events[0].bindings),
                KISEvent(
                    id="E2",
                    text="Woman speaking with chef wearing black",
                    images=[KISImageRef(asset_id="ast_chef_1", content_type="image/jpeg")],
                    bindings=res_5.intent.events[1].bindings,
                ),
                KISEvent(id="E3", text="Woman taking white plate", bindings=res_5.intent.events[2].bindings),
            ],
            temporal_edges=res_5.intent.temporal_edges,
        )
        global_rewriter.rewrite.return_value = intent_rev5

        req_6 = KISSearchRequest(
            base_intent=res_5.intent,
            expected_revision=4,
            operation=GlobalRewriteOperation(
                kind="global_rewrite",
                instruction="resolve all references explicitly",
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        res_6 = service.search_kis(req_6)
        self.assertEqual(res_6.intent.revision, 5)
        self.assertEqual([e.id for e in res_6.intent.events], ["E1", "E2", "E3"])
        # E2 still preserves image ref
        self.assertEqual(res_6.intent.events[1].images[0].asset_id, "ast_chef_1")

        # ---------------------------------------------------------------------
        # 7. SearchOnly: change retrieval parameters without semantic mutation -> still Rev 5
        # ---------------------------------------------------------------------
        req_7 = KISSearchRequest(
            base_intent=res_6.intent,
            expected_revision=5,
            operation=SearchOnlyOperation(kind="search_only"),
            use_dense=True,
            use_bm25=False,
            top_k=50,
        )
        res_7 = service.search_kis(req_7)
        self.assertEqual(res_7.intent.revision, 5)
        self.assertEqual(res_7.operation_summary.kind, "search_only")
        call_7 = service.kis.execute.call_args[1]
        self.assertEqual(call_7["top_k"], 50)
        self.assertTrue(call_7["use_dense"])
        self.assertFalse(call_7["use_bm25"])

        # ---------------------------------------------------------------------
        # 8. Force semantic provider failure on later patch -> base Rev 5 remains safe
        # ---------------------------------------------------------------------
        scoped_resolver.resolve.side_effect = InferenceResponseError("Provider unavailable")

        req_8 = KISSearchRequest(
            base_intent=res_6.intent,
            expected_revision=5,
            operation=PatchEventsOperation(
                kind="patch_events",
                patches=[EventPatch(event_id="E2", instruction="chef is wearing white")],
            ),
            use_dense=True,
            use_bm25=True,
            top_k=20,
        )
        with self.assertRaises(InferenceResponseError):
            service.search_kis(req_8)

        # Baseline Rev 5 remains unaltered
        self.assertEqual(res_6.intent.revision, 5)
        self.assertEqual(res_6.intent.events[1].images[0].asset_id, "ast_chef_1")

    def test_baseline_config_gives_positive_weight_to_visual_image(self) -> None:
        """Verify configs/baseline.yaml assigns positive weight to visual_image."""
        config_path = Path("configs/baseline.yaml")
        self.assertTrue(config_path.exists(), "configs/baseline.yaml must exist")
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        adaptive_weights = cfg["search"]["hybrid_temporal"]["adaptive"]["base_component_weights"]
        self.assertIn("visual_image", adaptive_weights)
        self.assertGreater(adaptive_weights["visual_image"], 0.0)

    def test_image_only_kis_search_plan_preserves_images_and_no_exploration_seed(self) -> None:
        """Verify image-only request creates valid retrieval plan and response has no exploration seed."""
        img_intent = KISIntent(
            revision=1,
            query_text=None,
            entities=[],
            events=[
                KISEvent(
                    id="E1",
                    text=None,
                    images=[KISImageRef(asset_id="ast_query_photo", content_type="image/jpeg")],
                    bindings=[],
                )
            ],
            temporal_edges=[],
        )

        service = self._make_service()
        service.kis.execute.return_value = self.mock_execution

        req = KISSearchRequest(
            base_intent=img_intent,
            expected_revision=1,
            operation=SearchOnlyOperation(kind="search_only"),
            use_dense=True,
            use_bm25=True,
            top_k=10,
        )
        res = service.search_kis(req)

        plan = service.kis.execute.call_args.kwargs["retrieval_plan"]
        self.assertEqual(len(plan.events), 1)
        self.assertEqual(plan.events[0].event_id, "E1")
        self.assertIsNone(plan.events[0].canonical_text)
        self.assertEqual(len(plan.events[0].image_refs), 1)
        self.assertEqual(plan.events[0].image_refs[0].asset_id, "ast_query_photo")
        self.assertFalse(hasattr(res, "exploration_seed"))

    def test_retrieval_plan_projects_direct_multilingual_text_and_preserves_images(self) -> None:
        """Verify multilingual text is directly projected to dense and bm25 without translation, and image refs preserved."""
        mixed_intent = KISIntent(
            revision=1,
            query_text="Một người phụ nữ đứng, một đầu bếp nấu ăn.",
            entities=[],
            events=[
                KISEvent(id="E1", text="Một người phụ nữ đứng", bindings=[]),
                KISEvent(
                    id="E2",
                    text=None,
                    images=[KISImageRef(asset_id="ast_pan", content_type="image/png")],
                    bindings=[],
                ),
                KISEvent(id="E3", text="Một đầu bếp nấu ăn", bindings=[]),
            ],
            temporal_edges=[
                KISTemporalEdge(source="E1", target="E2"),
                KISTemporalEdge(source="E2", target="E3"),
            ],
        )

        service = self._make_service()
        service.kis.execute.return_value = self.mock_execution

        req = KISSearchRequest(
            base_intent=mixed_intent,
            expected_revision=1,
            operation=SearchOnlyOperation(kind="search_only"),
            use_dense=True,
            use_bm25=True,
            top_k=10,
        )
        service.search_kis(req)

        # Retrieval plan preserved exact order, direct multilingual text, and image refs
        call = service.kis.execute.call_args[1]
        plan = call["retrieval_plan"]
        self.assertEqual(plan.event_ids, ("E1", "E2", "E3"))

        self.assertEqual(plan.events[0].event_id, "E1")
        self.assertEqual(plan.events[0].canonical_text, "Một người phụ nữ đứng")
        self.assertEqual(plan.events[0].dense_text, "Một người phụ nữ đứng")
        self.assertEqual(plan.events[0].bm25_text, "Một người phụ nữ đứng")
        self.assertEqual(len(plan.events[0].image_refs), 0)

        self.assertEqual(plan.events[1].event_id, "E2")
        self.assertIsNone(plan.events[1].canonical_text)
        self.assertIsNone(plan.events[1].dense_text)
        self.assertIsNone(plan.events[1].bm25_text)
        self.assertEqual(len(plan.events[1].image_refs), 1)
        self.assertEqual(plan.events[1].image_refs[0].asset_id, "ast_pan")

        self.assertEqual(plan.events[2].event_id, "E3")
        self.assertEqual(plan.events[2].canonical_text, "Một đầu bếp nấu ăn")
        self.assertEqual(plan.events[2].dense_text, "Một đầu bếp nấu ăn")
        self.assertEqual(plan.events[2].bm25_text, "Một đầu bếp nấu ăn")
        self.assertEqual(len(plan.events[2].image_refs), 0)

