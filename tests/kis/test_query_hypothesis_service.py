"""Tests for QueryHypothesisService lifecycle, preview, commit, undo, and revision guards."""

import pytest

from hcmai.kis.hypothesis.models import EditEvent, QueryHypothesisError
from hcmai.kis.hypothesis.service import QueryHypothesisService
from hcmai.kis.hypothesis.store import QueryHypothesisStore
from hcmai.kis.models import (
    KISEvent,
    KISIntent,
    SourceProvenance,
)


class FakeResolver:
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
def service() -> QueryHypothesisService:
    store = QueryHypothesisStore(ttl_seconds=1800, max_entries=50)
    return QueryHypothesisService(resolver=FakeResolver(), store=store)


@pytest.fixture
def opened_session(service: QueryHypothesisService):
    return service.open("người đàn ông ngồi xuống")


def test_preview_does_not_bump_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    before = service.get(opened_session.session_id)
    preview = service.preview(
        opened_session.session_id,
        expected_revision=before.intent.revision,
        action=EditEvent(event_id="E1", text="preview text"),
    )
    after = service.get(opened_session.session_id)
    assert preview.intent.revision == before.intent.revision + 1
    assert after.intent == before.intent


def test_commit_rejects_stale_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    service.commit(
        opened_session.session_id, 1, EditEvent(event_id="E1", text="first")
    )
    try:
        service.commit(
            opened_session.session_id, 1, EditEvent(event_id="E1", text="stale")
        )
    except QueryHypothesisError as exc:
        assert exc.code == "QUERY_REVISION_CONFLICT"
    else:
        raise AssertionError("stale query commit must fail")


def test_undo_creates_new_monotonic_revision(
    service: QueryHypothesisService, opened_session
) -> None:
    committed = service.commit(
        opened_session.session_id, 1, EditEvent(event_id="E1", text="changed")
    )
    restored = service.undo(opened_session.session_id, committed.intent.revision)
    assert restored.intent.revision == committed.intent.revision + 1
    assert restored.intent.events[0].text == "người đàn ông ngồi xuống"


def test_search_uses_server_hypothesis_not_client_replacement(service, opened_session):
    from unittest.mock import Mock
    from hcmai.api.contracts.kis import KISSearchRequest
    from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
    from hcmai.orchestration.pipeline import SearchService
    from hcmai.orchestration.workflows.kis import KISSearchExecution

    mock_execution = KISSearchExecution(
        results=[
            SearchResult(
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
        latency=SearchLatency(query_ms=5.0, retrieval_ms=10.0),
    )

    search_service = SearchService(
        corpus=Mock(),
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=FakeResolver(),
        query_hypotheses=service,
    )
    search_service.kis = Mock()
    search_service.kis.execute.return_value = mock_execution

    request = KISSearchRequest(
        query_hypothesis_session_id=opened_session.session_id,
        base_intent=None,
        expected_revision=opened_session.intent.revision,
        operation={"kind": "search_only"},
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    response = search_service.search_kis(request)
    assert response.intent == opened_session.intent
    assert response.query_hypothesis_session_id == opened_session.session_id
    call_args = search_service.kis.execute.call_args
    assert call_args.kwargs["intent"] == opened_session.intent


def test_search_binds_evidence_snapshot_to_hypothesis_revision(service, opened_session):
    from unittest.mock import Mock
    import numpy as np
    from hcmai.api.contracts.kis import KISSearchRequest
    from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
    from hcmai.event_trail.models import AlignedPath, DecoderConfigSnapshot, EvidenceSnapshot, VideoEventScores
    from hcmai.orchestration.workflows.search.temporal import TemporalSearchArtifact, TemporalSearchResult
    from hcmai.orchestration.pipeline import SearchService
    from hcmai.orchestration.workflows.kis import KISSearchExecution

    path = AlignedPath(
        video_id="v1",
        score=0.95,
        frame_ids=("v1_f1",),
        frame_idxs=(10,),
        timestamps_ms=(1000,),
    )
    res = TemporalSearchResult(
        paths=(path,),
        retrieval_ms=10.0,
        alignment_ms=5.0,
    )
    scores = (
        VideoEventScores(
            video_id="v1",
            frame_ids=np.array(["v1_f1"]),
            frame_idx=np.array([10]),
            timestamps_ms=np.array([1000], dtype=np.int64),
            scores=np.array([[0.95]], dtype=np.float32),
        ),
    )
    config = DecoderConfigSnapshot(
        lambda_gap=0.5,
        event_power=1.0,
        cluster_delta=2.0,
        path_min_separation_ms=1000,
    )
    artifact = TemporalSearchArtifact(
        result=res,
        video_scores=scores,
        decoder_config=config,
        scoring_revision="rev-local",
    )

    mock_execution = KISSearchExecution(
        results=[
            SearchResult(
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
        latency=SearchLatency(query_ms=5.0, retrieval_ms=10.0),
        temporal_artifact=artifact,
    )

    search_service = SearchService(
        corpus=Mock(),
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=FakeResolver(),
        query_hypotheses=service,
    )
    search_service.kis = Mock()
    search_service.kis.execute.return_value = mock_execution

    request = KISSearchRequest(
        query_hypothesis_session_id=opened_session.session_id,
        base_intent=None,
        expected_revision=opened_session.intent.revision,
        operation={"kind": "search_only"},
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    response = search_service.search_kis(request)
    assert response.evidence_snapshot_id is not None
    snapshot = search_service.event_trail_snapshots.get(response.evidence_snapshot_id)
    assert isinstance(snapshot, EvidenceSnapshot)
    assert snapshot.kis_revision == opened_session.intent.revision


def test_search_rejects_hypothesis_revision_conflict(service, opened_session):
    from unittest.mock import Mock
    from hcmai.api.contracts.kis import KISSearchRequest
    from hcmai.orchestration.utils.errors import RevisionConflictError
    from hcmai.orchestration.pipeline import SearchService

    search_service = SearchService(
        corpus=Mock(),
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=FakeResolver(),
        query_hypotheses=service,
    )

    request = KISSearchRequest(
        query_hypothesis_session_id=opened_session.session_id,
        base_intent=None,
        expected_revision=999,
        operation={"kind": "search_only"},
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    with pytest.raises(RevisionConflictError):
        search_service.search_kis(request)


def test_search_rejects_non_search_only_operation_with_hypothesis_session(service, opened_session):
    from unittest.mock import Mock
    from hcmai.api.contracts.kis import KISSearchRequest
    from hcmai.orchestration.utils.errors import InvalidQueryInputError
    from hcmai.orchestration.pipeline import SearchService

    search_service = SearchService(
        corpus=Mock(),
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=FakeResolver(),
        query_hypotheses=service,
    )

    request = KISSearchRequest(
        query_hypothesis_session_id=opened_session.session_id,
        base_intent=None,
        expected_revision=opened_session.intent.revision,
        operation={"kind": "global_rewrite", "instruction": "something"},
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )
    with pytest.raises(InvalidQueryInputError):
        search_service.search_kis(request)


def test_query_commit_log_contains_revisions(caplog, service, opened_session):
    import logging
    caplog.set_level(logging.INFO)
    service.commit(opened_session.session_id, 1, EditEvent(event_id="E1", text="edited"))
    records = [
        r
        for r in caplog.records
        if "query_hypothesis_event" in r.message
    ]
    assert len(records) > 0
    message = records[0].message
    assert '"query_revision": 2' in message
    assert '"action_type": "edit"' in message
    assert '"committed": true' in message


def test_query_preview_log_marks_not_committed(caplog, service, opened_session):
    import logging
    caplog.set_level(logging.INFO)
    service.preview(
        opened_session.session_id, 1, EditEvent(event_id="E1", text="preview text")
    )
    records = [
        r
        for r in caplog.records
        if "query_hypothesis_event" in r.message
    ]
    assert len(records) > 0
    message = records[0].message
    assert '"committed": false' in message
    assert '"query_revision": 1' in message


