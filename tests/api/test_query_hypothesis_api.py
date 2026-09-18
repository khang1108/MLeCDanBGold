"""Tests for KIS Query Hypothesis API routes, preview, commit, undo, and revision conflicts."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from hcmai.app import create_app
from hcmai.kis.hypothesis.service import QueryHypothesisService
from hcmai.kis.hypothesis.store import QueryHypothesisStore
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    SourceProvenance,
)


class StubResolver:
    def resolve_initial(self, text: str, revision: int = 1) -> KISIntent:
        return KISIntent(
            revision=revision,
            query_text=text,
            language="vi",
            entities=[],
            events=[
                KISEvent(
                    id="E1",
                    text=text,
                    origin="source",
                    source_provenance=SourceProvenance(
                        source_text=text, start_char=0, end_char=len(text)
                    ),
                    bindings=[],
                )
            ],
            temporal_edges=[],
        )


@pytest.fixture
def mock_search_service():
    service = Mock()
    store = QueryHypothesisStore(ttl_seconds=1800, max_entries=50)
    service.query_hypothesis_store = store
    service.query_hypotheses = QueryHypothesisService(
        store=store,
        resolver=StubResolver(),
    )
    service.llm = None
    service.health = Mock(return_value={"capabilities": {"search": True}})
    return service


@pytest.fixture
def client(mock_search_service) -> TestClient:
    app = create_app(search_service=mock_search_service)
    return TestClient(app)


@pytest.fixture
def seeded_query_hypothesis(client: TestClient) -> dict:
    resp = client.post(
        "/api/v1/kis/hypotheses/open",
        json={"text": "người đàn ông ngồi xuống"},
    )
    assert resp.status_code == 200
    return resp.json()


def test_stale_commit_returns_409(
    client: TestClient, seeded_query_hypothesis: dict
) -> None:
    session_id = seeded_query_hypothesis["session_id"]
    payload = {
        "expected_query_revision": 0,
        "action": {"type": "edit", "event_id": "E1", "text": "x"},
    }
    response = client.post(
        f"/api/v1/kis/hypotheses/{session_id}/commit", json=payload
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "QUERY_REVISION_CONFLICT"


def test_open_preview_commit_and_undo_endpoints(
    client: TestClient, seeded_query_hypothesis: dict
) -> None:
    session_id = seeded_query_hypothesis["session_id"]

    # 1. Preview
    prev_payload = {
        "expected_query_revision": 1,
        "action": {"type": "edit", "event_id": "E1", "text": "preview edit"},
    }
    prev_resp = client.post(
        f"/api/v1/kis/hypotheses/{session_id}/preview", json=prev_payload
    )
    assert prev_resp.status_code == 200
    assert prev_resp.json()["base_revision"] == 1
    assert prev_resp.json()["intent"]["events"][0]["text"] == "preview edit"

    # Verify session state was not mutated by preview
    get_resp = client.get(f"/api/v1/kis/hypotheses/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["query_revision"] == 1

    # 2. Commit
    commit_payload = {
        "expected_query_revision": 1,
        "action": {"type": "edit", "event_id": "E1", "text": "committed edit"},
    }
    commit_resp = client.post(
        f"/api/v1/kis/hypotheses/{session_id}/commit", json=commit_payload
    )
    assert commit_resp.status_code == 200
    assert commit_resp.json()["query_revision"] == 2
    assert commit_resp.json()["can_undo"] is True

    # 3. Undo
    undo_payload = {"expected_query_revision": 2}
    undo_resp = client.post(
        f"/api/v1/kis/hypotheses/{session_id}/undo", json=undo_payload
    )
    assert undo_resp.status_code == 200
    assert undo_resp.json()["query_revision"] == 3
    assert (
        undo_resp.json()["intent"]["events"][0]["text"]
        == "người đàn ông ngồi xuống"
    )


def test_search_kis_with_query_hypothesis_session_id(
    client: TestClient, mock_search_service, seeded_query_hypothesis: dict
) -> None:
    session_id = seeded_query_hypothesis["session_id"]
    from hcmai.api.contracts.kis import (
        KISOperationSummary,
        KISSearchResponse,
        KISSearchResult,
    )
    from hcmai.api.contracts.search import SearchLatency, SearchResultMetadata

    mock_search_service.search_kis.return_value = KISSearchResponse(
        intent=mock_search_service.query_hypotheses.get(session_id).intent,
        operation_summary=KISOperationSummary(
            kind="search_only", affected_event_ids=[]
        ),
        use_dense=True,
        use_bm25=False,
        results=[
            KISSearchResult(
                result_id="r_123",
                frame_id="v1_f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.9,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(query_ms=1.0, retrieval_ms=2.0),
        query_hypothesis_session_id=session_id,
    )

    resp = client.post(
        "/api/v1/kis/search",
        json={
            "query_hypothesis_session_id": session_id,
            "expected_revision": 1,
            "operation": {"kind": "search_only"},
            "use_dense": True,
            "use_bm25": False,
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["query_hypothesis_session_id"] == session_id
    call_req = mock_search_service.search_kis.call_args[0][0]
    assert call_req.query_hypothesis_session_id == session_id

