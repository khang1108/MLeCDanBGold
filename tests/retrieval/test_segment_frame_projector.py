"""Deterministic tests for projecting timeline ASR evidence to canonical frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hcmai.data.stores.frame import FrameStore
from hcmai.retrieval.retriever.segment.projector import SegmentFrameProjector


def _frame(
    frame_id: str,
    timestamp_ms: int,
    frame_idx: int,
    *,
    video_id: str = "v1",
) -> dict[str, object]:
    """Return one valid canonical frame row for projector fixtures."""

    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "image_path": f"/frames/{frame_id}.jpg",
        "width": 640,
        "height": 360,
    }


@pytest.fixture
def frame_store(tmp_path: Path) -> FrameStore:
    """Load deliberately unsorted rows to exercise the cached video ordering."""

    pytest.importorskip("pyarrow")
    path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            _frame("f_after", 2_500, 25),
            _frame("f_inside", 1_600, 16),
            _frame("f_before", 500, 5),
            _frame("f_tie_later", 1_700, 17),
            _frame("other", 1_500, 15, video_id="v2"),
        ]
    ).to_parquet(path, index=False)
    return FrameStore(path)


def test_frame_store_get_by_video_returns_cached_sorted_tuple(
    frame_store: FrameStore,
) -> None:
    """Expose the immutable per-video index without copying or re-sorting it."""

    first = frame_store.get_by_video("v1")
    second = frame_store.get_by_video("v1")

    assert first is second
    assert isinstance(first, tuple)
    assert [frame.frame_id for frame in first] == [
        "f_before",
        "f_inside",
        "f_tie_later",
        "f_after",
    ]
    assert frame_store.get_by_video("unknown") == ()


def test_projector_prefers_frame_inside_half_open_segment(
    frame_store: FrameStore,
) -> None:
    """Choose an in-span canonical frame before considering nearest fallback."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=5_000)

    projection = projector.project("v1", start_ms=1_000, end_ms=1_700)

    assert projection is not None
    assert projection.frame_id == "f_inside"
    assert projection.video_id == "v1"
    assert projection.frame_idx == 16
    assert projection.timestamp_ms == 1_600
    assert projection.distance_ms == 250
    assert projection.kind == "inside_segment"


def test_projector_excludes_frame_at_half_open_end(
    frame_store: FrameStore,
) -> None:
    """A frame exactly at ``end_ms`` is not inside ``[start_ms, end_ms)``."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=0)

    assert projector.project("v1", start_ms=1_699, end_ms=1_700) is None


def test_projector_uses_deterministic_midpoint_ties(tmp_path: Path) -> None:
    """Resolve equal midpoint distances by timestamp, frame index, then ID."""

    pytest.importorskip("pyarrow")
    path = tmp_path / "ties.parquet"
    pd.DataFrame(
        [
            _frame("f_later", 1_600, 16),
            _frame("f_z", 1_400, 15),
            _frame("f_a", 1_400, 15),
            _frame("f_lower_idx", 1_400, 14),
        ]
    ).to_parquet(path, index=False)
    projector = SegmentFrameProjector(
        FrameStore(path), max_projection_gap_ms=1_000
    )

    projection = projector.project("v1", start_ms=1_000, end_ms=2_000)

    assert projection is not None
    assert projection.frame_id == "f_lower_idx"
    assert projection.kind == "inside_segment"


def test_projector_rejects_far_midpoint_fallback(
    frame_store: FrameStore,
) -> None:
    """Do not attach distant timeline evidence to an unrelated keyframe."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=100)

    assert projector.project("v1", start_ms=9_000, end_ms=10_000) is None


def test_projector_accepts_fallback_at_inclusive_gap_boundary(
    frame_store: FrameStore,
) -> None:
    """The configured maximum gap is inclusive for nearest-midpoint fallback."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=500)

    projection = projector.project("v1", start_ms=2_900, end_ms=3_100)

    assert projection is not None
    assert projection.frame_id == "f_after"
    assert projection.distance_ms == 500
    assert projection.kind == "nearest_midpoint"


def test_project_row_and_unknown_video_are_safe(frame_store: FrameStore) -> None:
    """Project mapping rows directly and skip segments without canonical video data."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=500)

    assert projector.project("", start_ms=1_000, end_ms=2_000) is None
    assert projector.project("unknown", start_ms=1_000, end_ms=2_000) is None
    projection = projector.project_row(
        {"video_id": "v1", "start_ms": 1_000, "end_ms": 2_000}
    )
    assert projection is not None
    assert projection.frame_id == "f_inside"


@pytest.mark.parametrize(
    ("start_ms", "end_ms"),
    [(-1, 1), (2, 2), (3, 2), (1.5, 2), (1, 2.5)],
)
def test_projector_rejects_invalid_intervals(
    frame_store: FrameStore, start_ms: object, end_ms: object
) -> None:
    """Reject malformed public intervals rather than coercing timeline identity."""

    projector = SegmentFrameProjector(frame_store, max_projection_gap_ms=500)

    with pytest.raises(ValueError, match="segment interval"):
        projector.project("v1", start_ms=start_ms, end_ms=end_ms)  # type: ignore[arg-type]


@pytest.mark.parametrize("gap", [-1, 1.5, True])
def test_projector_rejects_invalid_maximum_gap(
    frame_store: FrameStore, gap: object
) -> None:
    """Require a non-negative integer projection distance budget."""

    with pytest.raises(ValueError, match="max_projection_gap_ms"):
        SegmentFrameProjector(frame_store, max_projection_gap_ms=gap)  # type: ignore[arg-type]
