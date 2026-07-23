from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from hcmai.common.schemas import FrameRecord
from hcmai.data import FrameStore, prepare_frames


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "btc"
    mappings = root / "map-keyframes-aic25-b1" / "map-keyframes"
    keyframes = root / "Keyframes_L21" / "keyframes"
    mappings.mkdir(parents=True)
    rows = {
        "L21_V002": [(3, 0.3, 700), (1, 0.1, 700)],
        "L21_V001": [(2, 0.2, 333), (1, 0.1, 90)],
    }
    for video_id, values in rows.items():
        table = pd.DataFrame(values, columns=["n", "pts_time", "frame_idx"])
        table.to_csv(mappings / f"{video_id}.csv", index=False)
        directory = keyframes / video_id
        directory.mkdir(parents=True)
        for order, _time, _official_idx in reversed(values):
            Image.new("RGB", (8, 6)).save(directory / f"{order:03d}.jpg")
    return root


def test_build_is_canonical_and_store_resolves_official_mapping(
    dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "frames.parquet"
    command = [sys.executable, "scripts/prepare_data.py", "--dataset-root",
               str(dataset), "--output", str(output)]
    completed = subprocess.run(
        command, cwd=Path(__file__).parents[1], capture_output=True, text=True
    )
    assert completed.returncode == 0 and "Status: PASSED" in completed.stdout
    first = pd.read_parquet(output)
    prepare_frames(dataset, output)
    second = pd.read_parquet(output)
    assert list(output.parent.iterdir()) == [output]
    assert list(first.columns) == [
        "frame_id", "video_id", "frame_idx", "keyframe_order",
        "timestamp_ms", "image_path", "width", "height",
    ]
    assert first.equals(second)
    assert first[["video_id", "keyframe_order"]].values.tolist() == [
        ["L21_V001", 1], ["L21_V001", 2], ["L21_V002", 1],
        ["L21_V002", 3]]
    assert first["frame_idx"].tolist() == [90, 333, 700, 700]
    assert all(not Path(value).is_absolute() for value in first["image_path"])
    rows = first.to_dict(orient="records")
    assert all(FrameRecord.model_validate(row) for row in rows)
    store = FrameStore.load(output)
    record = store.get("L21_V001_keyframe_000001")
    assert (record.video_id, record.frame_idx) == ("L21_V001", 90)
    assert [row.frame_id for row in store.iter_frames()] == (
        first["frame_id"].tolist()
    )
    assert store.contains_submission("L21_V001", 90)
    assert not store.contains_submission("L21_V001", 1)


def test_missing_image_fails_without_overwriting(
    dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "frames.parquet"
    prepare_frames(dataset, output)
    original = output.read_bytes()
    image = dataset / "Keyframes_L21/keyframes/L21_V001/001.jpg"
    image.unlink()
    with pytest.raises(ValueError, match="no image"):
        prepare_frames(dataset, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("value", [None, -1, 1.5])
def test_invalid_frame_idx_fails(dataset: Path, tmp_path: Path, value) -> None:
    mapping = dataset / "map-keyframes-aic25-b1/map-keyframes/L21_V001.csv"
    table = pd.read_csv(mapping)
    table["frame_idx"] = table["frame_idx"].astype(object)
    table.loc[0, "frame_idx"] = value
    table.to_csv(mapping, index=False)
    with pytest.raises(ValueError, match="frame_idx|required"):
        prepare_frames(dataset, tmp_path / "frames.parquet")


def test_duplicate_frame_id_fails(dataset: Path, tmp_path: Path) -> None:
    mapping = dataset / "map-keyframes-aic25-b1/map-keyframes/L21_V001.csv"
    table = pd.read_csv(mapping)
    table.loc[1, "n"] = table.loc[0, "n"]
    table.to_csv(mapping, index=False)
    with pytest.raises(ValueError, match="Duplicate frame_id"):
        prepare_frames(dataset, tmp_path / "frames.parquet")
