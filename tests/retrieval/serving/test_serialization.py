"""Tests for HTTP serving serialization helpers."""

import numpy as np
import pytest

from hcmai.orchestration.workflows.temporal_search import (
    DecoderConfigSnapshot,
    TemporalSearchArtifact,
    TemporalSearchResult,
)
from hcmai.retrieval.plan import (
    KISImageRef,
    KISRetrievalEvent,
    KISRetrievalPlan,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.retrieval.serving.utils.serialization import (
    artifact_to_schema,
    path_to_schema,
    plan_to_schema,
    schema_to_artifact,
    schema_to_path,
    schema_to_plan,
    schema_to_video_scores,
    video_scores_to_schema,
)
from hcmai.temporal.dp import AlignedPath
from tests.retrieval.serving.fakes import make_fake_corpus


def test_plan_serialization_roundtrip() -> None:
    event1 = KISRetrievalEvent(
        event_id="E1",
        canonical_text="a red car",
        dense_text="red vehicle",
        bm25_text="car red",
        image_refs=(KISImageRef(asset_id="asset-1", content_type="image/jpeg"),),
    )
    event2 = KISRetrievalEvent(
        event_id="E2",
        canonical_text="turns left",
        dense_text="turn left",
        bm25_text=None,
        image_refs=(),
    )
    plan = KISRetrievalPlan(events=(event1, event2))

    schema = plan_to_schema(plan, use_dense=True, use_bm25=False, top_k=15)
    assert schema.top_k == 15
    assert len(schema.events) == 2
    assert schema.events[0].event_id == "E1"
    assert schema.events[0].image_asset_ids == ["asset-1"]

    restored = schema_to_plan(schema)
    assert len(restored.events) == 2
    assert restored.events[0].event_id == "E1"
    assert restored.events[0].canonical_text == "a red car"
    assert restored.events[0].dense_text == "red vehicle"
    assert restored.events[0].bm25_text == "car red"
    assert len(restored.events[0].image_refs) == 1
    assert restored.events[0].image_refs[0].asset_id == "asset-1"
    assert restored.events[1].dense_text == "turn left"


def test_video_scores_serialization_roundtrip() -> None:
    video = VideoEventScores(
        video_id="video-1",
        frame_ids=np.array(["v1_f1", "v1_f2"]),
        frame_idx=np.array([10, 20], dtype=np.int64),
        timestamps_ms=np.array([1000, 2000], dtype=np.int64),
        scores=np.array([[0.5, 0.8]], dtype=np.float32),
    )
    corpus = make_fake_corpus()

    schema = video_scores_to_schema(video)
    assert schema.video_id == "video-1"
    assert np.allclose(schema.scores, [[0.5, 0.8]])

    restored = schema_to_video_scores(schema, corpus=corpus)
    assert restored.video_id == "video-1"
    np.testing.assert_array_equal(restored.frame_ids, video.frame_ids)
    np.testing.assert_array_equal(restored.frame_idx, video.frame_idx)
    np.testing.assert_array_equal(restored.timestamps_ms, video.timestamps_ms)
    np.testing.assert_allclose(restored.scores, video.scores)


def test_temporal_search_artifact_roundtrip() -> None:
    path = AlignedPath(
        video_id="video-1",
        score=0.92,
        frame_ids=("v1_f1",),
        frame_idxs=(10,),
        timestamps_ms=(1000,),
    )
    result = TemporalSearchResult(
        paths=(path,),
        retrieval_ms=12.5,
        alignment_ms=3.2,
    )
    video = VideoEventScores(
        video_id="video-1",
        frame_ids=np.array(["v1_f1"]),
        frame_idx=np.array([10], dtype=np.int64),
        timestamps_ms=np.array([1000], dtype=np.int64),
        scores=np.array([[0.92]], dtype=np.float32),
    )
    decoder_config = DecoderConfigSnapshot(
        lambda_gap=0.2,
        event_power=1.1,
        cluster_delta=0.8,
        path_min_separation_ms=1500,
    )
    artifact = TemporalSearchArtifact(
        result=result,
        video_scores=(video,),
        decoder_config=decoder_config,
        scoring_revision="rev-test-123",
    )
    corpus = make_fake_corpus()

    schema = artifact_to_schema(artifact)
    assert schema.scoring_revision == "rev-test-123"
    assert len(schema.result.paths) == 1

    restored = schema_to_artifact(schema, corpus=corpus)
    assert restored.scoring_revision == "rev-test-123"
    assert len(restored.result.paths) == 1
    assert restored.result.paths[0].video_id == "video-1"
    assert restored.result.paths[0].score == 0.92
    assert restored.decoder_config.lambda_gap == 0.2
