"""Task 2 seed projection regressions for selected-video scoring."""

from unittest.mock import Mock

import pytest

from hcmai.orchestration.workflows.temporal_exploration import QueryBinding, TemporalExploration
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from orchestration.test_temporal_exploration import _TemporalSearch


@pytest.mark.parametrize("use_dense,use_bm25", [(True, True), (True, False), (False, True)])
def test_task2_step8_binding_scores_the_committed_plan(use_dense, use_bm25):
    temporal = _TemporalSearch()
    temporal.score_videos = Mock(wraps=temporal.score_videos)
    binding = QueryBinding(
        retrieval_plan=KISRetrievalPlan(events=(KISRetrievalEvent("E1", "canonical", "translated" if use_dense else None, "literal" if use_bm25 else None),)),
        semantic_revision=3,
        event_version="events-v1",
        scoring_revision="scores-v1",
        use_dense=use_dense,
        use_bm25=use_bm25,
    )
    TemporalExploration(temporal).open(binding, "video-1", (0, 100))
    temporal.score_videos.assert_called_once_with(
        ("canonical",), retrieval_events=("translated",) if use_dense else ("canonical",),
        caption_events=("literal",) if use_bm25 else None,
        use_dense=use_dense, use_bm25=use_bm25,
    )


def test_task2_step8_binding_rejects_no_textual_scoring_source_before_scoring():
    temporal = Mock()
    binding = QueryBinding(
        retrieval_plan=KISRetrievalPlan(events=(KISRetrievalEvent("E1", "canonical", None, None),)),
        semantic_revision=1, event_version="events-v1", scoring_revision="scores-v1",
        use_dense=True, use_bm25=False,
    )
    with pytest.raises(ValueError):
        TemporalExploration(temporal).open(binding, "video-1", (0, 100))
    temporal.score_videos.assert_not_called()
