"""Tests for TRAKE projection of shared canonical temporal-search paths."""

from __future__ import annotations

import pytest

from hcmai.api.contracts import TRAKERequest
from hcmai.orchestration.temporal_search import TemporalSearchResult
from hcmai.orchestration.workflows.trake import TRAKEPipeline
from hcmai.temporal import AlignedPath


class FakeAlignment:
    """Return fixed same-video paths and retain the supplied event sequence."""

    def __init__(self) -> None:
        """Initialize a call log for the shared-service boundary assertion."""

        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search(
        self,
        events: list[str],
        *,
        top_k: int,
    ) -> TemporalSearchResult:
        """Return two independently ranked paths from the same video."""

        self.calls.append((tuple(events), top_k))
        return TemporalSearchResult(
            paths=(
                AlignedPath(
                    video_id="v1",
                    score=2.41,
                    frame_ids=("f 0", "f1"),
                    frame_idxs=(10, 20),
                    timestamps_ms=(1_000, 2_000),
                ),
                AlignedPath(
                    video_id="v1",
                    score=2.11,
                    frame_ids=("f2", "f3"),
                    frame_idxs=(30, 40),
                    timestamps_ms=(3_000, 4_000),
                ),
            ),
            retrieval_ms=12.5,
            alignment_ms=7.25,
        )


def test_trake_keeps_same_video_paths_independent() -> None:
    """Project every ranked path without merging equal-video alignments."""

    alignment = FakeAlignment()
    response = TRAKEPipeline(alignment).execute(
        TRAKERequest(events=["e1", "e2"], top_k=2)
    )

    assert response.events == ["e1", "e2"]
    assert len(response.paths) == 2
    assert [path.video_id for path in response.paths] == ["v1", "v1"]
    assert response.paths[0].frame_ids != response.paths[1].frame_ids
    assert alignment.calls == [(("e1", "e2"), 2)]


def test_trake_preserves_ordered_arrays_and_raw_scores() -> None:
    """Expose canonical path values without rewriting their identity."""

    response = TRAKEPipeline(FakeAlignment()).execute(
        TRAKERequest(events=["e1", "e2"], top_k=2)
    )
    path = response.paths[0]

    assert path.score == pytest.approx(2.41)
    assert path.frame_ids == ["f 0", "f1"]
    assert path.frame_idxs == [10, 20]
    assert path.timestamps_ms == [1_000, 2_000]
    assert len(path.frame_ids) == len(response.events)
    assert len(path.frame_idxs) == len(response.events)
    assert len(path.timestamps_ms) == len(response.events)
    assert response.latency.retrieval_ms == 12.5
    assert response.latency.alignment_ms == 7.25
    assert response.latency.materialization_ms >= 0
    assert response.latency.total_ms >= response.latency.materialization_ms


def test_trake_returns_empty_paths_for_valid_unalignable_events() -> None:
    """Keep no-path outcomes successful rather than fabricating a submission."""

    class EmptyAlignment:
        """Return a valid empty temporal-search result."""

        def search(self, events: list[str], *, top_k: int) -> TemporalSearchResult:
            """Provide no alignment while retaining non-negative stage timings."""

            del events, top_k
            return TemporalSearchResult(
                paths=(),
                retrieval_ms=1.0,
                alignment_ms=2.0,
            )

    response = TRAKEPipeline(EmptyAlignment()).execute(
        TRAKERequest(events=["e1", "e2"])
    )

    assert response.paths == []
