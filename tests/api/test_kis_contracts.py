"""Contract validation tests for simplified KIS search contracts."""

import unittest
import pytest
from pydantic import ValidationError

from hcmai.api.contracts.kis import (
    KISOperationSummary,
    KISSearchRequest,
    KISSearchResponse,
    KISSearchResult,
)
from hcmai.kis.models import (
    KISEvent,
    KISImageRef,
    KISIntent,
    KISTemporalEdge,
)


def _base_intent(with_image: bool = False) -> KISIntent:
    images = [KISImageRef(asset_id="img_123", content_type="image/jpeg")] if with_image else []
    return KISIntent(
        revision=2,
        query_text="woman enters, then talks",
        entities=[],
        events=[
            KISEvent(id="E1", text="A woman enters.", images=images, bindings=[]),
            KISEvent(id="E2", text="The woman talks.", images=[], bindings=[]),
        ],
        temporal_edges=[KISTemporalEdge(source="E1", target="E2", relation="before")],
    )


class KISContractsTest(unittest.TestCase):
    def test_search_with_base_intent(self) -> None:
        base = _base_intent()
        request = KISSearchRequest.model_validate({
            "base_intent": base.model_dump(),
            "expected_revision": base.revision,
            "use_dense": True,
            "use_bm25": False,
            "top_k": 50,
        })
        self.assertIsNotNone(request.base_intent)
        self.assertEqual(request.top_k, 50)
        self.assertEqual(request.expected_revision, 2)

    def test_search_with_query_hypothesis_session(self) -> None:
        request = KISSearchRequest.model_validate({
            "query_hypothesis_session_id": "sess_abc123",
            "expected_revision": 1,
            "use_dense": True,
            "use_bm25": True,
        })
        self.assertEqual(request.query_hypothesis_session_id, "sess_abc123")
        self.assertIsNone(request.base_intent)

    def test_search_request_requires_target(self) -> None:
        with self.assertRaises(ValidationError):
            KISSearchRequest.model_validate({
                "base_intent": None,
                "query_hypothesis_session_id": None,
                "expected_revision": 0,
                "use_dense": True,
            })

    def test_search_request_requires_at_least_one_retrieval_source(self) -> None:
        base = _base_intent(with_image=False)
        with self.assertRaises(ValidationError):
            KISSearchRequest.model_validate({
                "base_intent": base.model_dump(),
                "expected_revision": 2,
                "use_dense": False,
                "use_bm25": False,
            })

    def test_image_evidence_allows_text_sources_disabled(self) -> None:
        base = _base_intent(with_image=True)
        request = KISSearchRequest.model_validate({
            "base_intent": base.model_dump(),
            "expected_revision": 2,
            "use_dense": False,
            "use_bm25": False,
            "top_k": 20,
        })
        self.assertFalse(request.use_dense)
        self.assertFalse(request.use_bm25)


def test_kis_search_response_structure() -> None:
    from hcmai.api.contracts.latency import SearchLatency

    assert "dense_events" not in KISSearchResponse.model_fields
    assert "bm25_events" not in KISSearchResponse.model_fields
    assert "exploration_seed" not in KISSearchResponse.model_fields
    assert "operation_summary" in KISSearchResponse.model_fields
    assert SearchLatency(intent_ms=1, translation_ms=2).translation_ms == 2


def test_kis_search_result_requires_result_id() -> None:
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
