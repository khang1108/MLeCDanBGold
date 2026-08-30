"""Tests for authoritative organizer keyframe-coordinate mapping."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from offline.ingestion.keyframe_map import (
    join_btc_mapping,
    load_btc_keyframe_map,
    project_keyframe_paths,
)


def _write_mapping(root: Path, rows: list[dict[str, object]]) -> None:
    """Write one synthetic organizer map for the default fixture video."""

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "L01_V001.csv", index=False)


def _source_frames() -> pd.DataFrame:
    """Return two source rows with stable internal identity."""

    return pd.DataFrame(
        [
            {"frame_id": "f1", "video_id": "L01_V001", "keyframe_order": 1},
            {"frame_id": "f2", "video_id": "L01_V001", "keyframe_order": 2},
        ]
    )


def test_mapping_preserves_exact_btc_coordinates(tmp_path: Path) -> None:
    """Use organizer FPS and timestamps exactly without standard-FPS snapping."""

    root = tmp_path / "map_keyframes"
    _write_mapping(
        root,
        [
            {"n": 1, "pts_time": 1.001, "fps": 29.97, "frame_idx": 30},
            {"n": 2, "pts_time": 2.002, "fps": 29.97, "frame_idx": 60},
        ],
    )

    joined = join_btc_mapping(_source_frames(), load_btc_keyframe_map(root))

    assert joined["fps"].tolist() == [29.97, 29.97]
    assert joined["frame_idx"].tolist() == [30, 60]
    assert joined["timestamp_ms"].tolist() == [1001, 2002]


def test_duplicate_submission_coordinates_are_allowed(tmp_path: Path) -> None:
    """Keep separate internal frames when BTC reuses a frame_idx."""

    root = tmp_path / "map_keyframes"
    _write_mapping(
        root,
        [
            {"n": 1, "pts_time": 0.040, "fps": 25.0, "frame_idx": 1},
            {"n": 2, "pts_time": 0.079, "fps": 25.0, "frame_idx": 1},
        ],
    )

    joined = join_btc_mapping(_source_frames(), load_btc_keyframe_map(root))

    assert joined["frame_id"].tolist() == ["f1", "f2"]
    assert joined["frame_idx"].tolist() == [1, 1]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"n": 1, "pts_time": 0.0, "fps": 25.0, "frame_idx": 0},
                {"n": 3, "pts_time": 0.1, "fps": 25.0, "frame_idx": 2},
            ],
            "contiguous",
        ),
        (
            [
                {"n": 1, "pts_time": 0.0, "fps": 25.0, "frame_idx": 0},
                {"n": 2, "pts_time": 0.1, "fps": 30.0, "frame_idx": 3},
            ],
            "one positive value",
        ),
        ([{"n": 1, "pts_time": -0.1, "fps": 25.0, "frame_idx": 0}], "non-negative"),
    ],
)
def test_mapping_rejects_invalid_coordinate_series(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    """Reject mapping rows that cannot be a canonical organizer coordinate map."""

    root = tmp_path / "map_keyframes"
    _write_mapping(root, rows)

    with pytest.raises(ValueError, match=message):
        load_btc_keyframe_map(root)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("n", 1.5, "finite integral"),
        ("n", np.inf, "finite integral"),
        ("n", np.nan, "finite integral"),
        ("frame_idx", 1.5, "finite integral"),
        ("frame_idx", np.inf, "finite integral"),
        ("frame_idx", np.nan, "finite integral"),
        ("pts_time", np.inf, "finite numeric"),
        ("pts_time", np.nan, "finite numeric"),
        ("fps", np.inf, "finite numeric"),
        ("fps", np.nan, "finite numeric"),
    ],
)
def test_mapping_rejects_fractional_or_non_finite_values(
    tmp_path: Path, column: str, value: float, message: str
) -> None:
    """Reject malformed organizer values before any integer cast can alter them."""

    root = tmp_path / "map_keyframes"
    row = {"n": 1, "pts_time": 0.0, "fps": 25.0, "frame_idx": 0}
    row[column] = value
    _write_mapping(root, [row])

    with pytest.raises(ValueError, match=message):
        load_btc_keyframe_map(root)


def test_join_rejects_source_frame_missing_from_mapping(tmp_path: Path) -> None:
    """Reject metadata that has no organizer coordinate for a keyframe."""

    root = tmp_path / "map_keyframes"
    _write_mapping(root, [{"n": 1, "pts_time": 0.0, "fps": 25.0, "frame_idx": 0}])

    with pytest.raises(ValueError, match="missing from BTC mapping"):
        join_btc_mapping(_source_frames(), load_btc_keyframe_map(root))


def test_join_rejects_fractional_submission_coordinate() -> None:
    """Defend direct callers from truncating an unvalidated frame_idx."""

    source = _source_frames().iloc[:1]
    mapping = pd.DataFrame(
        [
            {
                "video_id": "L01_V001",
                "keyframe_order": 1,
                "pts_time": 0.0,
                "fps": 25.0,
                "frame_idx": 1.5,
            }
        ]
    )

    with pytest.raises(ValueError, match="finite integral"):
        join_btc_mapping(source, mapping)


def test_project_keyframe_paths_uses_portable_staged_images(tmp_path: Path) -> None:
    """Replace only image paths when remote workers stage BTC keyframes."""

    keyframes_root = tmp_path / "keyframes"
    staged = keyframes_root / "L01_V001"
    staged.mkdir(parents=True)
    (staged / "b.png").touch()
    (staged / "a.jpg").touch()
    frames = pd.DataFrame(
        [
            {
                "frame_id": "f2",
                "video_id": "L01_V001",
                "keyframe_order": 2,
                "frame_idx": 60,
                "timestamp_ms": 2002,
                "image_path": "/stale/b.jpg",
            },
            {
                "frame_id": "f1",
                "video_id": "L01_V001",
                "keyframe_order": 1,
                "frame_idx": 30,
                "timestamp_ms": 1001,
                "image_path": "/stale/a.jpg",
            },
        ]
    )

    projected = project_keyframe_paths(frames, keyframes_root)

    assert projected["image_path"].tolist() == [str(staged / "b.png"), str(staged / "a.jpg")]
    assert projected[["frame_id", "frame_idx", "timestamp_ms"]].to_dict("records") == frames[["frame_id", "frame_idx", "timestamp_ms"]].to_dict("records")
    assert frames["image_path"].tolist() == ["/stale/b.jpg", "/stale/a.jpg"]
