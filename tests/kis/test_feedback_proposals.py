"""Tests for de-authorizing generic chat and returning semantic proposals."""

from unittest.mock import Mock
from uuid import uuid4
import pytest

from hcmai.api.contracts.feedback import (
    FeedbackOpenRequest,
    FeedbackTurnRequest,
)
from hcmai.api.contracts.kis import KISSearchResult
from hcmai.api.contracts.search import SearchResultMetadata
from hcmai.kis.feedback.models import (
    QueryEditProposalAction,
    RefineRetrievalAction,
)
from hcmai.kis.feedback.resolver import FeedbackResolver
from hcmai.kis.feedback.service import FeedbackService
from hcmai.kis.feedback.store import FeedbackSessionStore
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    KISTemporalEdge,
    SourceProvenance,
)


def _make_sample_intent() -> KISIntent:
    return KISIntent(
        revision=1,
        query_text="Người đàn ông bước vào phòng rồi ngồi xuống",
        language="vi",
        entities=[],
        events=[
            KISEvent(
                id="E1",
                text="Người đàn ông bước vào phòng",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="Người đàn ông bước vào phòng",
                    start_char=0,
                    end_char=27,
                ),
                bindings=[],
            ),
            KISEvent(
                id="E2",
                text="rồi ngồi xuống",
                origin="source",
                source_provenance=SourceProvenance(
                    source_text="rồi ngồi xuống",
                    start_char=28,
                    end_char=42,
                ),
                bindings=[],
            ),
        ],
        temporal_edges=[
            KISTemporalEdge(source="E1", target="E2", relation="before"),
        ],
    )


class StubSearchExecution:
    def __init__(self, results: list[KISSearchResult], snapshot_id: str = "snap_test"):
        self.results = results
        self.evidence_snapshot_id = snapshot_id


class StubSearchService:
    def __init__(self):
        self.searches_run: list[dict] = []
        self.event_trail_snapshots = Mock()
        self.event_trail_snapshots.get.return_value = Mock(
            snapshot_id="snap_test",
            results={},
        )

    def execute_search(self, intent, retrieval_plan, use_dense=True, use_bm25=True, top_k=20, exclusions=None):
        self.searches_run.append({
            "intent": intent,
            "plan": retrieval_plan,
        })
        res = [
            KISSearchResult(
                result_id="r_1",
                frame_id="v1_f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.9,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ]
        return StubSearchExecution(results=res)


@pytest.fixture
def feedback_service_and_session():
    store = FeedbackSessionStore()
    search_service = StubSearchService()
    resolver = Mock(spec=FeedbackResolver)
    service = FeedbackService(store=store, resolver=resolver, search_service=search_service)

    opened = service.open(
        FeedbackOpenRequest(
            intent=_make_sample_intent(),
            original_query="Người đàn ông bước vào phòng rồi ngồi xuống",
            evidence_snapshot_id="snap_test",
        )
    )
    return service, opened, resolver, search_service


def test_chat_split_request_returns_proposal_without_mutating_intent(feedback_service_and_session):
    service, session, resolver, search_service = feedback_service_and_session
    before_intent = session.intent

    # Resolver resolves to QueryEditProposalAction
    resolver.resolve.return_value = QueryEditProposalAction(
        action={"type": "split", "event_id": "E1", "split_at": 15, "image_assignments": {}},
        explanation="Tách E1 thành 2 sự kiện",
    )

    req = FeedbackTurnRequest(
        request_id="req_split",
        expected_feedback_revision=session.feedback_revision,
        expected_kis_revision=session.intent.revision,
        message="tách E1 ra làm 2",
    )
    response = service.turn(session.session_id, req)

    # Intent is untouched, revision not bumped
    assert response.intent == before_intent
    assert response.status == "proposal"
    assert response.query_proposal is not None
    assert response.query_proposal["action"]["type"] == "split"
    # Search was NOT run for proposal
    assert len(search_service.searches_run) == 0


def test_refine_retrieval_still_changes_only_retrieval_override(feedback_service_and_session):
    service, session, resolver, search_service = feedback_service_and_session

    resolver.resolve.return_value = RefineRetrievalAction(
        event_ids=["E1"],
        refinements={"E1": "man in blue shirt"},
    )

    req = FeedbackTurnRequest(
        request_id="req_refine",
        expected_feedback_revision=session.feedback_revision,
        expected_kis_revision=session.intent.revision,
        message="tìm áo xanh cho E1",
    )
    response = service.turn(session.session_id, req)

    assert response.intent == session.intent
    assert "E1" in response.retrieval_overrides
    assert response.retrieval_overrides["E1"].dense_text == "man in blue shirt"
    assert len(search_service.searches_run) == 1
