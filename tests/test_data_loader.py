"""Tests for indexed access to canonical frame metadata."""

from pathlib import Path

import pandas as pd
import pytest

from hcmai.common.schemas.frame import FrameRecord
from hcmai.common.schemas.search import SearchFilters
from hcmai.data import loader
from hcmai.data.loader import FrameStore


@pytest.fixture
def frame_rows() -> list[dict[str, object]]:
    """Return deliberately unordered metadata from two videos."""

    return [
        {
            "frame_id": "L21_V001_00000030",
            "video_id": "L21_V001",
            "frame_idx": 30,
            "timestamp_ms": 1_000,
            "image_path": "/data/keyframes/L21_V001/3.jpg",
            "thumbnail_path": "/data/thumbnails/L21_V001/3.jpg",
            "width": 1_920,
            "height": 1_080,
        },
        {
            "frame_id": "L21_V001_00000010",
            "video_id": "L21_V001",
            "frame_idx": 10,
            "timestamp_ms": 400,
            "image_path": "/data/keyframes/L21_V001/1.jpg",
            "thumbnail_path": None,
            "width": 1_920,
            "height": 1_080,
        },
        {
            "frame_id": "L21_V001_00000020",
            "video_id": "L21_V001",
            "frame_idx": 20,
            "timestamp_ms": 1_000,
            "image_path": "/data/keyframes/L21_V001/2.jpg",
            "thumbnail_path": "/data/thumbnails/L21_V001/2.jpg",
            "width": 1_920,
            "height": 1_080,
        },
        {
            "frame_id": "L21_V001_00000040",
            "video_id": "L21_V001",
            "frame_idx": 40,
            "timestamp_ms": 1_600,
            "image_path": "/data/keyframes/L21_V001/4.jpg",
            "thumbnail_path": "/data/thumbnails/L21_V001/4.jpg",
            "width": 1_920,
            "height": 1_080,
        },
        {
            "frame_id": "L21_V002_00000005",
            "video_id": "L21_V002",
            "frame_idx": 5,
            "timestamp_ms": 700,
            "image_path": "/data/keyframes/L21_V002/1.jpg",
            "thumbnail_path": "/data/thumbnails/L21_V002/1.jpg",
            "width": 1_280,
            "height": 720,
        },
    ]


@pytest.fixture
def metadata_path(
    tmp_path: Path,
    frame_rows: list[dict[str, object]],
) -> Path:
    """Write the frame rows to a small Parquet fixture."""

    path = tmp_path / "frames.parquet"
    pd.DataFrame(frame_rows).to_parquet(path, index=False)
    return path


def test_store_loads_parquet_once(
    metadata_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load metadata only during construction, not during lookups."""

    original_read_parquet = loader.pd.read_parquet
    calls = 0

    def read_parquet_spy(path: Path) -> pd.DataFrame:
        """Count reads while delegating to pandas."""

        nonlocal calls
        calls += 1
        return original_read_parquet(path)

    monkeypatch.setattr(loader.pd, "read_parquet", read_parquet_spy)

    store = FrameStore(metadata_path)
    store.get("L21_V001_00000030")
    store.get_many(["L21_V001_00000010", "L21_V001_00000020"])
    store.filter_frame_ids(None)

    assert calls == 1


def test_get_returns_frame_record_with_nullable_values(
    metadata_path: Path,
) -> None:
    """Materialize Parquet rows as validated frame contracts."""

    frame = FrameStore(metadata_path).get("L21_V001_00000010")

    assert isinstance(frame, FrameRecord)
    assert frame.frame_idx == 10
    assert frame.thumbnail_path is None


def test_get_reports_unknown_frame_with_context(metadata_path: Path) -> None:
    """Include both the missing ID and source path in lookup errors."""

    store = FrameStore(metadata_path)

    with pytest.raises(KeyError) as error:
        store.get("missing-frame")

    message = str(error.value)
    assert "missing-frame" in message
    assert str(metadata_path) in message


def test_get_many_preserves_order_and_duplicates(metadata_path: Path) -> None:
    """Return one result for every requested ID in the same order."""

    store = FrameStore(metadata_path)
    requested = [
        "L21_V002_00000005",
        "L21_V001_00000020",
        "L21_V002_00000005",
    ]

    assert [frame.frame_id for frame in store.get_many(requested)] == requested


def test_get_neighbors_stays_in_video_and_sorts_by_time(
    metadata_path: Path,
) -> None:
    """Sort inclusive-window neighbors and exclude the target by default."""

    store = FrameStore(metadata_path)

    neighbors = store.get_neighbors(
        "L21_V001_00000030",
        window_ms=600,
    )

    assert [frame.frame_id for frame in neighbors] == [
        "L21_V001_00000010",
        "L21_V001_00000020",
        "L21_V001_00000040",
    ]


def test_get_neighbors_can_include_target(metadata_path: Path) -> None:
    """Include the target in timestamp and frame-index order when requested."""

    store = FrameStore(metadata_path)

    neighbors = store.get_neighbors(
        "L21_V001_00000030",
        window_ms=0,
        include_self=True,
    )

    assert [frame.frame_id for frame in neighbors] == [
        "L21_V001_00000020",
        "L21_V001_00000030",
    ]


def test_get_neighbors_rejects_negative_window(metadata_path: Path) -> None:
    """Reject a temporal window that cannot describe a valid range."""

    with pytest.raises(ValueError, match="window_ms"):
        FrameStore(metadata_path).get_neighbors(
            "L21_V001_00000030",
            window_ms=-1,
        )


def test_filter_frame_ids_applies_inclusive_supported_filters(
    metadata_path: Path,
) -> None:
    """Filter on video and time while deliberately ignoring min_score."""

    store = FrameStore(metadata_path)
    filters = SearchFilters(
        video_ids=["L21_V001"],
        start_time_ms=1_000,
        end_time_ms=1_600,
        min_score=10.0,
    )

    assert store.filter_frame_ids(filters) == [
        "L21_V001_00000030",
        "L21_V001_00000020",
        "L21_V001_00000040",
    ]


def test_filter_frame_ids_without_filters_preserves_metadata_order(
    metadata_path: Path,
    frame_rows: list[dict[str, object]],
) -> None:
    """Return every frame in canonical metadata order without restrictions."""

    expected = [str(row["frame_id"]) for row in frame_rows]
    store = FrameStore(metadata_path)

    assert store.filter_frame_ids(None) == expected
    assert store.filter_frame_ids(SearchFilters()) == expected
