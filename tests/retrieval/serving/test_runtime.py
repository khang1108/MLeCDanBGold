"""Tests for standalone RetrievalRuntime composition and delegation."""

from unittest.mock import Mock
import pytest

from hcmai.kis.models import KISImageRef
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.serving.runtime import RetrievalRuntime
from tests.retrieval.serving.fakes import (
    make_fake_corpus,
    make_fake_image_scorer,
    make_fake_temporal_search_service,
)


@pytest.fixture
def corpus():
    return make_fake_corpus()


@pytest.fixture
def temporal():
    return make_fake_temporal_search_service()


@pytest.fixture
def image_scorer():
    return make_fake_image_scorer()


@pytest.fixture
def image_search():
    service = Mock()
    service.search.return_value = Mock(results=[], latency=Mock(query_ms=1.0, retrieval_ms=2.0))
    return service


@pytest.fixture
def plan():
    img = KISImageRef(asset_id="sha256:test", content_type="image/jpeg")
    return KISRetrievalPlan(
        events=(
            KISRetrievalEvent("E1", "person walking", "person walking", "person walking", image_refs=(img,)),
        )
    )


@pytest.fixture
def runtime(corpus, temporal, image_scorer, image_search):
    return RetrievalRuntime(
        corpus=corpus,
        temporal=temporal,
        image_scorer=image_scorer,
        image_search=image_search,
        active_modalities=("visual", "context", "bm25"),
        startup_messages=(),
        max_temporal_event_count=5,
        image_max_upload_bytes=10 * 1024 * 1024,
        image_max_pixels=4096 * 4096,
        scoring_revision="rev-test-123",
    )


def test_REQ_005_runtime_scores_kis_images_before_existing_temporal_search(runtime, plan) -> None:
    artifact = runtime.search_plan(plan, use_dense=True, use_bm25=True, top_k=5)

    runtime.image_scorer.score_events.assert_called_once_with(plan.image_ref_rows)
    runtime.temporal.search_plan_artifact.assert_called_once_with(
        plan,
        image_component=runtime.image_scorer.score_events.return_value,
        use_dense=True,
        use_bm25=True,
        top_k=5,
    )
    assert artifact.scoring_revision == runtime.scoring_revision


def test_REQ_010_runtime_score_video_returns_one_video(runtime, plan) -> None:
    selected = runtime.score_video(plan, "video-2", use_dense=True, use_bm25=False)
    assert selected.video.video_id == "video-2"
    assert selected.scoring_revision == runtime.scoring_revision


def test_capabilities_report_modalities_without_running_retrieval(runtime) -> None:
    caps = runtime.capabilities()
    assert caps.ready is True
    assert caps.scoring_revision == "rev-test-123"
    assert caps.active_modalities == ("visual", "context", "bm25")
    assert caps.max_temporal_event_count == 5
    assert caps.image_max_upload_bytes == 10 * 1024 * 1024
    assert caps.image_max_pixels == 4096 * 4096

    runtime.temporal.search_plan_artifact.assert_not_called()
    runtime.temporal.search.assert_not_called()
    runtime.temporal.score_video.assert_not_called()
