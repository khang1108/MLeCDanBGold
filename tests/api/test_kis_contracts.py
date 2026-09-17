"""Contract validation tests for stateless semantic KIS operations."""

import unittest
import pytest
from pydantic import ValidationError

from hcmai.api.contracts.kis import (
    EventPatch,
    GlobalRewriteOperation,
    InitialResolveOperation,
    KISOperationSummary,
    KISSearchRequest,
    KISSearchResponse,
    PatchEventsOperation,
    SearchOnlyOperation,
)
from hcmai.kis.models import (
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)


def _base_intent() -> KISIntent:
    return KISIntent(
        revision=2,
        query_text="woman enters, then talks",
        entities=[],
        events=[
            KISEvent(id="E1", text="A woman enters.", images=[], bindings=[]),
            KISEvent(id="E2", text="The woman talks.", images=[], bindings=[]),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2", relation="before")],
    )


class KISContractsTest(unittest.TestCase):
    def test_initial_resolve_natural_text(self) -> None:
        request = KISSearchRequest.model_validate({
            "base_intent": None,
            "expected_revision": 0,
            "operation": {
                "kind": "initial_resolve",
                "text": "woman enters kitchen",
                "image_refs": [],
            },
            "use_dense": True,
            "use_bm25": True,
            "top_k": 20,
        })
        self.assertIsNone(request.base_intent)
        self.assertEqual(request.expected_revision, 0)
        self.assertEqual(request.operation.kind, "initial_resolve")
        self.assertEqual(request.operation.text, "woman enters kitchen")

    def test_initial_resolve_rejects_both_natural_and_explicit_patches(self) -> None:
        with self.assertRaises(ValidationError):
            InitialResolveOperation(
                kind="initial_resolve",
                text="natural text",
                patches=[EventPatch(event_id="E1", instruction="chef")],
            )

    def test_initial_resolve_rejects_neither_natural_nor_explicit(self) -> None:
        with self.assertRaises(ValidationError):
            InitialResolveOperation(kind="initial_resolve")

    def test_search_only_operation(self) -> None:
        base = _base_intent()
        request = KISSearchRequest.model_validate({
            "base_intent": base.model_dump(),
            "expected_revision": base.revision,
            "operation": {"kind": "search_only"},
            "use_dense": True,
            "use_bm25": False,
            "top_k": 100,
        })
        self.assertIsNotNone(request.base_intent)
        self.assertEqual(request.operation.kind, "search_only")
        self.assertEqual(request.top_k, 100)

    def test_patch_events_operation(self) -> None:
        base = _base_intent()
        request = KISSearchRequest.model_validate({
            "base_intent": base.model_dump(),
            "expected_revision": base.revision,
            "operation": {
                "kind": "patch_events",
                "patches": [
                    {
                        "event_id": "E2",
                        "instruction": "The woman talks to a chef.",
                        "add_image_ids": ["sha256:abc"],
                        "remove_image_ids": [],
                    }
                ],
            },
            "use_dense": True,
            "use_bm25": True,
            "top_k": 10,
        })
        self.assertEqual(request.operation.kind, "patch_events")
        self.assertEqual(len(request.operation.patches), 1)
        self.assertEqual(request.operation.patches[0].event_id, "E2")

    def test_patch_events_rejects_empty_patches(self) -> None:
        with self.assertRaises(ValidationError):
            PatchEventsOperation(kind="patch_events", patches=[])

    def test_global_rewrite_operation(self) -> None:
        base = _base_intent()
        request = KISSearchRequest.model_validate({
            "base_intent": base.model_dump(),
            "expected_revision": base.revision,
            "operation": {
                "kind": "global_rewrite",
                "instruction": "Resolve all pronouns explicitly.",
            },
        })
        self.assertEqual(request.operation.kind, "global_rewrite")
        self.assertEqual(
            request.operation.instruction, "Resolve all pronouns explicitly."
        )

    def test_global_rewrite_rejects_blank_instruction(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalRewriteOperation(kind="global_rewrite", instruction="   ")

    def test_search_request_requires_at_least_one_retrieval_source(self) -> None:
        with self.assertRaises(ValidationError):
            KISSearchRequest.model_validate({
                "base_intent": None,
                "expected_revision": 0,
                "operation": {
                    "kind": "initial_resolve",
                    "text": "woman enters kitchen",
                },
                "use_dense": False,
                "use_bm25": False,
            })



# ---- Task 1: evidence-aware source validation regressions ----

def test_image_only_initial_request_allows_text_sources_disabled() -> None:
    """Image-only initial resolve must be accepted even when both text toggles are false."""
    request = KISSearchRequest.model_validate({
        "base_intent": None,
        "expected_revision": 0,
        "operation": {
            "kind": "initial_resolve",
            "image_refs": [{"asset_id": "img_abc", "content_type": "image/png"}],
        },
        "use_dense": False,
        "use_bm25": False,
        "top_k": 20,
    })
    assert request.use_dense is False
    assert request.use_bm25 is False


def test_text_only_initial_request_rejects_all_text_sources_disabled() -> None:
    """Text-only initial resolve must raise when both use_dense=False and use_bm25=False."""
    try:
        KISSearchRequest.model_validate({
            "base_intent": None,
            "expected_revision": 0,
            "operation": {"kind": "initial_resolve", "text": "woman enters"},
            "use_dense": False,
            "use_bm25": False,
        })
    except ValueError as exc:
        assert "retrieval source" in str(exc).lower()
    else:
        raise AssertionError("text-only request must require a text retrieval source")


def test_seed_response_removes_prepared_string_aliases() -> None:
    from hcmai.api.contracts.kis import KISExplorationSeed
    from hcmai.api.contracts.latency import SearchLatency

    seed = KISExplorationSeed(
        semantic_revision=1,
        events=[
            {
                "event_id": "E1",
                "canonical_text": "boat",
                "dense_text": "boat",
            }
        ],
        use_dense=True,
        use_bm25=False,
    )
    assert seed.model_dump()["events"][0]["dense_text"] == "boat"
    assert "dense_events" not in KISSearchResponse.model_fields
    assert "bm25_events" not in KISSearchResponse.model_fields
    assert "exploration_seed" in KISSearchResponse.model_fields
    assert "operation_summary" in KISSearchResponse.model_fields
    assert SearchLatency(intent_ms=1, translation_ms=2).translation_ms == 2


def test_kis_search_result_requires_result_id() -> None:
    from hcmai.api.contracts.kis import KISSearchResult
    from hcmai.api.contracts.search import SearchResultMetadata

    with pytest.raises(ValidationError):
        KISSearchResult.model_validate({
            "frame_id": "v1_f1",
            "video_id": "v1",
            "frame_idx": 10,
            "timestamp_ms": 1000,
            "score": 0.95,
            "frame_ids": ["v1_f1"],
            "timestamps_ms": [1000],
            "metadata": {},
        })

    valid = KISSearchResult.model_validate({
        "result_id": "r_abc123",
        "frame_id": "v1_f1",
        "video_id": "v1",
        "frame_idx": 10,
        "timestamp_ms": 1000,
        "score": 0.95,
        "frame_ids": ["v1_f1"],
        "timestamps_ms": [1000],
        "metadata": {},
    })
    assert valid.result_id == "r_abc123"
    assert "evidence_snapshot_id" in KISSearchResponse.model_fields
    assert "results" in KISSearchResponse.model_fields
