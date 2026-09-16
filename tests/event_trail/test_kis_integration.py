"""Tests for KIS search snapshot handoff into EventTrail evidence store."""

from unittest.mock import Mock
import numpy as np
import pytest

from hcmai.api.contracts.kis import InitialResolveOperation, KISSearchRequest
from hcmai.api.contracts.search import SearchLatency, SearchResult, SearchResultMetadata
from hcmai.corpus.models import Frame
from hcmai.event_trail.models import EvidenceSnapshot
from hcmai.kis.models import KISEvent, KISIntent
from hcmai.orchestration.pipeline import SearchService
from hcmai.orchestration.workflows.kis import KISSearchExecution
from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.temporal.dp import AlignedPath
from hcmai.retrieval.retriever.video_scores import VideoEventScores


@pytest.fixture
def corpus():
    c = Mock()
    c.frame.side_effect = lambda fid: Frame(
        video_id="v1",
        frame_id=fid,
        frame_idx=10,
        timestamp_ms=1000,
        image_path="/tmp/f.jpg",
    )
    return c


@pytest.fixture
def mock_artifact():
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
    return TemporalSearchArtifact(
        result=res,
        video_scores=scores,
        decoder_config=config,
    )


def test_search_kis_captures_snapshot_and_assigns_result_ids(corpus, mock_artifact):
    service = SearchService(
        corpus=corpus,
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=Mock(),
        scoped_resolver=Mock(),
        global_rewriter=Mock(),
        event_translator=Mock(),
        kis_image_assets=Mock(),
    )
    # Intent resolver returning 1 event
    intent = KISIntent(
        revision=1,
        language="en",
        query_text="woman enters",
        entities=[],
        events=[KISEvent(id="E1", text="woman enters", images=[], bindings=[])],
        temporal_edges=[],
    )
    service.intent_resolver.resolve_initial.return_value = intent

    mock_exec = KISSearchExecution(
        results=[
            SearchResult(
                frame_id="v1_f1",
                video_id="v1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.95,
                frame_ids=["v1_f1"],
                timestamps_ms=[1000],
                metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(retrieval_ms=10.0, alignment_ms=5.0),
        temporal_artifact=mock_artifact,
    )
    service.kis = Mock()
    service.kis.execute.return_value = mock_exec

    request = KISSearchRequest(
        base_intent=None,
        expected_revision=0,
        operation=InitialResolveOperation(
            kind="initial_resolve",
            text="woman enters",
        ),
        use_dense=True,
        use_bm25=False,
        top_k=5,
    )

    response = service.search_kis(request)
    assert response.evidence_snapshot_id is not None
    assert len(response.results) == 1
    assert response.results[0].result_id.startswith("r_")
    assert response.latency.snapshot_ms >= 0

    snapshot = service.event_trail_snapshots.get(response.evidence_snapshot_id)
    assert isinstance(snapshot, EvidenceSnapshot)
    assert set(snapshot.video_evidence) == {result.video_id for result in response.results}
    assert len(snapshot.results) == len(response.results)
    assert response.results[0].result_id in snapshot.results
    service.kis.execute.assert_called_once()


def test_search_kis_multiple_results_share_same_video_evidence(corpus):
    service = SearchService(
        corpus=corpus,
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=Mock(),
        scoped_resolver=Mock(),
        global_rewriter=Mock(),
        event_translator=Mock(),
        kis_image_assets=Mock(),
    )
    intent = KISIntent(
        revision=1,
        language="en",
        query_text="woman enters",
        entities=[],
        events=[KISEvent(id="E1", text="woman enters", images=[], bindings=[])],
        temporal_edges=[],
    )
    service.intent_resolver.resolve_initial.return_value = intent

    p1 = AlignedPath(video_id="v1", score=0.9, frame_ids=("v1_f1",), frame_idxs=(10,), timestamps_ms=(1000,))
    p2 = AlignedPath(video_id="v1", score=0.8, frame_ids=("v1_f2",), frame_idxs=(20,), timestamps_ms=(2000,))
    res = TemporalSearchResult(paths=(p1, p2), retrieval_ms=10.0, alignment_ms=5.0)
    scores = (
        VideoEventScores(
            video_id="v1",
            frame_ids=np.array(["v1_f1", "v1_f2"]),
            frame_idx=np.array([10, 20]),
            timestamps_ms=np.array([1000, 2000], dtype=np.int64),
            scores=np.array([[0.9, 0.8]], dtype=np.float32),
        ),
    )
    artifact = TemporalSearchArtifact(
        result=res,
        video_scores=scores,
        decoder_config=DecoderConfigSnapshot(0.5, 1.0, 2.0, 1000),
    )
    mock_exec = KISSearchExecution(
        results=[
            SearchResult(
                frame_id="v1_f1", video_id="v1", frame_idx=10, timestamp_ms=1000,
                score=0.9, frame_ids=["v1_f1"], timestamps_ms=[1000], metadata=SearchResultMetadata(),
            ),
            SearchResult(
                frame_id="v1_f2", video_id="v1", frame_idx=20, timestamp_ms=2000,
                score=0.8, frame_ids=["v1_f2"], timestamps_ms=[2000], metadata=SearchResultMetadata(),
            ),
        ],
        latency=SearchLatency(retrieval_ms=10.0, alignment_ms=5.0),
        temporal_artifact=artifact,
    )
    service.kis = Mock()
    service.kis.execute.return_value = mock_exec

    request = KISSearchRequest(
        base_intent=None,
        expected_revision=0,
        operation=InitialResolveOperation(kind="initial_resolve", text="woman enters"),
        use_dense=True,
        use_bm25=False,
    )

    response = service.search_kis(request)
    assert len(response.results) == 2
    snapshot = service.event_trail_snapshots.get(response.evidence_snapshot_id)
    # Exactly one VideoEventScores entry for "v1", referenced by both results
    assert len(snapshot.video_evidence) == 1
    assert "v1" in snapshot.video_evidence
    assert len(snapshot.results) == 2


def test_search_kis_degrades_gracefully_on_memory_error(corpus, mock_artifact, monkeypatch):
    service = SearchService(
        corpus=corpus,
        retrieval=Mock(),
        temporal_evidence=Mock(),
        intent_resolver=Mock(),
        scoped_resolver=Mock(),
        global_rewriter=Mock(),
        event_translator=Mock(),
        kis_image_assets=Mock(),
    )
    intent = KISIntent(
        revision=1,
        language="en",
        query_text="woman enters",
        entities=[],
        events=[KISEvent(id="E1", text="woman enters", images=[], bindings=[])],
        temporal_edges=[],
    )
    service.intent_resolver.resolve_initial.return_value = intent
    mock_exec = KISSearchExecution(
        results=[
            SearchResult(
                frame_id="v1_f1", video_id="v1", frame_idx=10, timestamp_ms=1000,
                score=0.95, frame_ids=["v1_f1"], timestamps_ms=[1000], metadata=SearchResultMetadata(),
            )
        ],
        latency=SearchLatency(retrieval_ms=10.0, alignment_ms=5.0),
        temporal_artifact=mock_artifact,
    )
    service.kis = Mock()
    service.kis.execute.return_value = mock_exec

    # Simulate MemoryError during freeze
    def mock_freeze(v):
        raise MemoryError("out of memory")

    monkeypatch.setattr("hcmai.orchestration.pipeline.freeze_video_scores", mock_freeze)

    request = KISSearchRequest(
        base_intent=None,
        expected_revision=0,
        operation=InitialResolveOperation(kind="initial_resolve", text="woman enters"),
        use_dense=True,
        use_bm25=False,
    )

    response = service.search_kis(request)
    assert response.evidence_snapshot_id is None
    assert "EVENT_TRAIL_UNAVAILABLE" in response.warnings
    assert response.latency.snapshot_ms >= 0

