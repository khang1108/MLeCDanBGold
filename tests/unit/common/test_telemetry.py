from __future__ import annotations

from dataclasses import asdict

import pytest

from hcmai.common.observability import RetrievalTrace, StageStatus, StageTrace


def _stage(stage: str, duration_ms: float) -> StageTrace:
    return StageTrace(
        stage=stage,
        started_at=1.0,
        ended_at=1.0 + duration_ms / 1_000,
        duration_ms=duration_ms,
        status=StageStatus.SUCCESS,
    )


def test_pipeline_trace_aggregates_prefixed_stages_without_mutation() -> None:
    visual = RetrievalTrace(stages={"query_encoding": _stage("query_encoding", 2)})
    caption = RetrievalTrace(
        stages={"query_encoding": _stage("query_encoding", 5)}
    )

    merged = visual.merged(caption, prefix="caption")

    assert list(visual.stages) == ["query_encoding"]
    assert merged.duration_for("query_encoding") == 7
    assert merged.total_duration_ms == 7
    assert asdict(merged)["stages"]["caption.query_encoding"]["stage"] == (
        "caption.query_encoding"
    )
    with pytest.raises(ValueError, match="duplicate"):
        visual.merged(caption)


def test_pipeline_trace_rejects_mismatched_stage_key() -> None:
    with pytest.raises(ValueError, match="must match"):
        RetrievalTrace(stages={"wrong": _stage("query_encoding", 2)})


def test_failed_stage_requires_an_error_category() -> None:
    with pytest.raises(ValueError, match="error_category"):
        StageTrace(
            stage="index_search",
            started_at=1,
            ended_at=2,
            duration_ms=1_000,
            status=StageStatus.FAILED,
        )
