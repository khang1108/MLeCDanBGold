"""Task 2 and Task 9 seed projection regressions for selected-video scoring."""

from unittest.mock import Mock
import numpy as np
import pytest

from hcmai.kis.models import KISImageRef
from hcmai.orchestration.workflows.temporal_exploration import QueryBinding, TemporalExploration
from hcmai.retrieval.evidence.components import TemporalScoreComponent
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
try:
    from tests.orchestration.test_temporal_exploration import _TemporalSearch
except ImportError:
    from orchestration.test_temporal_exploration import _TemporalSearch



@pytest.mark.parametrize("use_dense,use_bm25", [(True, True), (True, False), (False, True)])
def test_task2_step8_binding_scores_the_committed_plan(use_dense, use_bm25):
    temporal = _TemporalSearch()
    temporal.score_video = Mock(wraps=temporal.score_video)
    temporal.score_plan = Mock(wraps=temporal.score_plan)
    plan = KISRetrievalPlan(events=(KISRetrievalEvent("E1", "canonical", "translated" if use_dense else None, "literal" if use_bm25 else None),))
    binding = QueryBinding(
        retrieval_plan=plan,
        semantic_revision=3,
        event_version="events-v1",
        scoring_revision="scores-v1",
        use_dense=use_dense,
        use_bm25=use_bm25,
    )
    TemporalExploration(temporal).open(binding, "video-1", (0, 100))
    temporal.score_video.assert_called_once_with(
        plan,
        video_id="video-1",
        image_component=None,
        use_dense=use_dense,
        use_bm25=use_bm25,
    )
    temporal.score_plan.assert_not_called()


def test_task2_step8_binding_rejects_no_textual_scoring_source_before_scoring():
    temporal = Mock()
    binding = QueryBinding(
        retrieval_plan=KISRetrievalPlan(events=(KISRetrievalEvent("E1", "canonical", None, None),)),
        semantic_revision=1, event_version="events-v1", scoring_revision="scores-v1",
        use_dense=True, use_bm25=False,
    )
    with pytest.raises(ValueError):
        TemporalExploration(temporal).open(binding, "video-1", (0, 100))
    temporal.score_video.assert_not_called()
    temporal.score_plan.assert_not_called()


def test_task9_binding_opens_image_only_event():
    temporal = _TemporalSearch()
    temporal.score_video = Mock(wraps=temporal.score_video)
    temporal.score_plan = Mock(wraps=temporal.score_plan)
    image_scorer = Mock()
    image_scorer.score_events.return_value = TemporalScoreComponent(
        name="visual_image",
        raw_scores=np.zeros((1, 2), dtype=np.float32),
    )
    img = KISImageRef(asset_id="sha256:abc", content_type="image/png")
    plan = KISRetrievalPlan(events=(
        KISRetrievalEvent("E1", None, None, None, image_refs=(img,)),
    ))
    binding = QueryBinding(
        retrieval_plan=plan,
        semantic_revision=1,
        event_version="events-v1",
        scoring_revision="scores-v1",
        use_dense=True,
        use_bm25=False,
    )
    view = TemporalExploration(temporal, image_scorer=image_scorer).open(binding, "video-1", (0, 100))
    assert view.video_id == "video-1"
    assert view.status == "no_valid_path"
    image_scorer.score_events.assert_called_once_with(((img,),))
    temporal.score_video.assert_called_once()
    temporal.score_plan.assert_not_called()
