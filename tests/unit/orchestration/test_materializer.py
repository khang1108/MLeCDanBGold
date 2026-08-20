"""Regression coverage for public retrieval-response materialization."""

from __future__ import annotations

from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.orchestration.materializer import _build_scores


def test_context_score_is_exposed_only_in_diagnostics() -> None:
    """Context ranking remains a score diagnostic rather than response evidence text."""

    scores = _build_scores(
        RetrievalCandidate(
            frame_id="frame-1",
            source_scores={RetrievalSource.CONTEXT: 0.73},
            final_score=0.73,
        )
    )

    assert scores.context == 0.73
    assert "context" in scores.model_dump()
