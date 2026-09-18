"""End-to-end integration tests for HTTP retrieval serving."""

import httpx
import pytest

from hcmai.orchestration.workflows.search.temporal import AlignedPath
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.serving.client import RetrievalHttpClient
from hcmai.retrieval.serving.remote import RemoteTemporalSearchService
from hcmai.retrieval.serving.runtime import RetrievalRuntime
from hcmai.retrieval.serving.server import create_app
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from tests.retrieval.serving.fakes import (
    make_fake_corpus,
    make_fake_image_scorer,
    make_fake_temporal_search_service,
)


@pytest.fixture
def corpus():
    return make_fake_corpus()


@pytest.fixture
def integration_runtime(corpus):
    temporal = make_fake_temporal_search_service()
    image_scorer = make_fake_image_scorer()
    return RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=image_scorer,
        image_search=None,
        active_modalities=("visual", "context"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=4096 * 4096,
        scoring_revision="rev-integration-test",
    )


from fastapi.testclient import TestClient


@pytest.fixture
def client(integration_runtime):
    app = create_app(integration_runtime)
    settings = RetrievalClientSettings(target="testserver", timeout_seconds=5.0)
    c = RetrievalHttpClient(settings)
    c._client = TestClient(app, base_url="http://testserver")
    try:
        yield c
    finally:
        c.close()


def test_full_search_plan_roundtrip_preserves_canonical_identity(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "a red sports car", "red car", "car"),
        )
    )

    artifact = remote_temporal.search_plan_artifact(plan, top_k=2)

    assert len(artifact.result.paths) == 2
    path1 = artifact.result.paths[0]
    assert path1.video_id == "video-1"
    assert path1.frame_ids == ("v1_f1",)
    assert path1.frame_idxs == (10,)
    assert path1.timestamps_ms == (1000,)
    assert artifact.scoring_revision == "rev-integration-test"

    assert len(artifact.video_scores) == 2
    v1 = artifact.video_scores[0]
    assert v1.video_id == "video-1"
    assert list(v1.frame_ids) == ["v1_f1", "v1_f2"]
    assert list(v1.frame_idx) == [10, 20]
    assert list(v1.timestamps_ms) == [1000, 2000]


def test_score_video_roundtrip(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "a red sports car", "red car", "car"),
        )
    )

    selected = remote_temporal.score_video(plan, video_id="video-1")
    assert selected.video.video_id == "video-1"
    assert list(selected.video.frame_ids) == ["v1_f1", "v1_f2"]
    assert selected.scoring_revision == "rev-integration-test"


def test_search_events_roundtrip(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    result = remote_temporal.search(["event 1", "event 2"], top_k=5)
    assert len(result.paths) == 2
    assert result.paths[0].video_id == "video-1"
