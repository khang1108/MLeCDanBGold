"""Tests for remote temporal and image search HTTP adapters."""

import httpx
import numpy as np
import pytest

from hcmai.orchestration.workflows.search.image import (
    ImageQueryTooLargeError,
    InvalidImageQueryError,
)
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.serving.client import (
    RemoteImageCandidate,
    RemoteImageSearchResult,
    RetrievalHttpClient,
)
from hcmai.retrieval.serving.remote import (
    RemoteImageSearchService,
    RemoteTemporalSearchService,
)
from hcmai.retrieval.serving.runtime import RetrievalRuntime
from hcmai.retrieval.serving.server import create_app
from hcmai.retrieval.serving.utils.config import RetrievalClientSettings
from hcmai.retrieval.serving.utils.errors import RetrievalUnavailableError
from tests.retrieval.serving.fakes import (
    make_fake_corpus,
    make_fake_image_scorer,
    make_fake_temporal_search_service,
    make_fake_video_scores,
)


@pytest.fixture
def corpus():
    return make_fake_corpus()


@pytest.fixture
def fake_runtime(corpus):
    temporal = make_fake_temporal_search_service()
    image_scorer = make_fake_image_scorer()
    return RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=image_scorer,
        image_search=None,
        active_modalities=("visual", "context", "bm25"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=4096 * 4096,
        scoring_revision="rev-adapter-test",
    )


from fastapi.testclient import TestClient


@pytest.fixture
def client(fake_runtime):
    app = create_app(fake_runtime)
    settings = RetrievalClientSettings(target="testserver", timeout_seconds=5.0)
    c = RetrievalHttpClient(settings)
    c._client = TestClient(app, base_url="http://testserver")
    try:
        yield c
    finally:
        c.close()


def test_remote_temporal_service_delegates_search_plan(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "canonical text", "dense text", "literal text"),
        )
    )
    artifact = remote_temporal.search_plan_artifact(plan, top_k=5)
    assert artifact.result.paths
    assert artifact.scoring_revision == "rev-adapter-test"


def test_remote_temporal_service_score_video(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    plan = KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "canonical text", "dense text", "literal text"),
        )
    )
    selected = remote_temporal.score_video(plan, video_id="video-1")
    assert selected.video.video_id == "video-1"
    assert selected.scoring_revision == "rev-adapter-test"


def test_remote_temporal_service_get_scoring_revision(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    revision = remote_temporal.get_scoring_revision()
    assert revision == "rev-adapter-test"


def test_remote_temporal_service_get_scoring_revision_rejects_unready(corpus) -> None:
    settings = RetrievalClientSettings(target="127.0.0.1:59999", timeout_seconds=1.0, health_timeout_seconds=0.1)
    client = RetrievalHttpClient(settings)
    try:
        remote_temporal = RemoteTemporalSearchService(corpus, client)
        with pytest.raises(RetrievalUnavailableError):
            remote_temporal.get_scoring_revision()
    finally:
        client.close()


def test_remote_image_service_validates_input_locally(client, corpus) -> None:
    service = RemoteImageSearchService(
        corpus,
        client,
        max_upload_bytes=100,
        max_pixels=1000,
    )
    with pytest.raises(InvalidImageQueryError, match="media type"):
        service.search(b"payload", content_type="application/pdf", top_k=5)

    with pytest.raises(InvalidImageQueryError, match="empty"):
        service.search(b"", content_type="image/jpeg", top_k=5)

    with pytest.raises(ImageQueryTooLargeError, match="exceeds"):
        service.search(b"a" * 101, content_type="image/jpeg", top_k=5)

    with pytest.raises(ValueError, match="greater than zero"):
        service.search(b"payload", content_type="image/jpeg", top_k=0)


def test_remote_temporal_service_search_events(client, corpus) -> None:
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    result = remote_temporal.search(["event 1"], top_k=5)
    assert len(result.paths) > 0


def test_remote_temporal_service_decode_video(corpus) -> None:
    from unittest.mock import Mock

    client = Mock()
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    v1_scores = make_fake_video_scores("video-1", ["v1_f1", "v1_f2"])
    allowed = np.array([[True, True]], dtype=bool)
    paths = remote_temporal.decode_video(v1_scores, allowed=allowed)
    assert len(paths) == 1
    assert paths[0].video_id == "video-1"
    assert client.method_calls == []


def test_remote_temporal_service_decode_event_alternatives(corpus) -> None:
    from unittest.mock import Mock

    client = Mock()
    remote_temporal = RemoteTemporalSearchService(corpus, client)
    v1_scores = make_fake_video_scores("video-1", ["v1_f1", "v1_f2"])
    allowed = np.array([[True, True]], dtype=bool)
    conditioned = remote_temporal.decode_event_alternatives(
        v1_scores,
        allowed=allowed,
        focus_event_index=0,
        max_paths=2,
        min_separation_ms=0,
    )
    assert len(conditioned) > 0
    assert conditioned[0].path.video_id == "video-1"
    assert client.method_calls == []


def test_remote_image_service_search_delegates_and_materializes(corpus) -> None:
    from unittest.mock import Mock

    client = Mock()
    client.search_image.return_value = RemoteImageSearchResult(
        candidates=(
            RemoteImageCandidate(
                video_id="video-1",
                frame_id="v1_f1",
                frame_idx=10,
                timestamp_ms=1000,
                score=0.92,
            ),
        ),
        query_ms=12.0,
        retrieval_ms=8.0,
    )
    service = RemoteImageSearchService(
        corpus,
        client,
        max_upload_bytes=1000,
        max_pixels=1000,
    )
    response = service.search(b"valid_payload", content_type="image/jpeg", top_k=5)
    assert len(response.results) == 1
    assert response.results[0].video_id == "video-1"
    assert response.results[0].frame_idx == 10
    assert response.latency.query_ms == 12.0
    assert response.latency.retrieval_ms == 8.0
    assert response.latency.materialization_ms > 0
