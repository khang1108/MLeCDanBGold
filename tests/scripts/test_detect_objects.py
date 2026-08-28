"""CLI and pipeline tests for the YOLOE object detection entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from hcmai.data.enrichment.object_detection import (
    ObjectDetectionConfig,
    load_vocab,
    run_yoloe,
)
from scripts.detect_objects import parse_args


@pytest.mark.parametrize("option", ("--top-k", "--batch-size", "--limit"))
@pytest.mark.parametrize("value", ("0", "-1"))
def test_parse_args_rejects_non_positive_work_limits(option: str, value: str) -> None:
    """Reject values that would make slicing or batching silently do no work."""

    with pytest.raises(SystemExit) as error:
        parse_args([option, value])

    assert error.value.code == 2


@pytest.mark.parametrize("value", ("-0.01", "1.01", "nan", "inf", "-inf"))
def test_parse_args_rejects_invalid_confidence(value: str) -> None:
    """Reject confidence thresholds outside the finite YOLO unit interval."""

    with pytest.raises(SystemExit) as error:
        parse_args(["--min-confidence", value])

    assert error.value.code == 2


@pytest.mark.parametrize("value", ("0", "1"))
def test_parse_args_accepts_confidence_boundaries(value: str) -> None:
    """Allow both inclusive confidence boundaries supported by the CLI contract."""

    args = parse_args(["--min-confidence", value])

    assert args.min_confidence == float(value)


def test_parse_args_defers_work_defaults_to_prepare_config() -> None:
    """Keep YAML as the source of truth unless an explicit override is passed."""

    args = parse_args([])

    assert args.config == Path("configs/prepare.yaml")
    assert args.top_k is None
    assert args.batch_size is None
    assert args.limit is None
    assert args.min_confidence is None


class _FakeVector:
    """Small tensor-like value used to exercise sorting and top-k selection."""

    def __init__(self, values: list[object]) -> None:
        self.values = values

    def argsort(self, *, descending: bool) -> "_FakeVector":
        return _FakeVector(
            sorted(
                range(len(self.values)),
                key=lambda index: self.values[index],
                reverse=descending,
            )
        )

    def __getitem__(self, index: object) -> object:
        if isinstance(index, _FakeVector):
            index = index.values
        if isinstance(index, slice):
            return _FakeVector(self.values[index])
        if isinstance(index, list):
            return _FakeVector([self.values[item] for item in index])
        return self.values[index]  # type: ignore[index]

    def tolist(self) -> list[object]:
        return list(self.values)


class _FakeBoxes:
    """Tensor-shaped detection fields returned by the fake YOLOE result."""

    conf = _FakeVector([0.4, 0.9])
    cls = _FakeVector([1, 0])
    xyxy = _FakeVector([[0.0, 0.0, 10.0, 10.0], [2.0, 1.0, 18.0, 9.0]])

    def __len__(self) -> int:
        return len(self.conf.values)


class _FakeResult:
    """One deterministic YOLOE result with two sortable detections."""

    orig_shape = (10, 20)
    names = {0: "person", 1: "car"}
    boxes = _FakeBoxes()


class _FakeModel:
    """Capture model invocation without importing or downloading Ultralytics."""

    names = _FakeResult.names

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, images: list[str], **kwargs: object) -> list[_FakeResult]:
        self.calls.append({"images": images, **kwargs})
        return [_FakeResult() for _ in images]

    def get_text_pe(self, names: list[str]) -> list[str]:
        return names

    def set_classes(self, names: list[str], embeddings: list[str]) -> None:
        self.prompted = names


def test_run_yoloe_publishes_raw_and_canonical_artifacts(tmp_path: Path) -> None:
    """The new pipeline turns fake YOLOE results into the enrichment contract."""

    image = tmp_path / "keyframes/v1/0000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    frames = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "v1:0000",
                "video_id": "v1",
                "frame_idx": 7,
                "timestamp_ms": 500,
                "image_path": "keyframes/v1/0000.jpg",
                "width": 20,
                "height": 10,
            }
        ]
    ).to_parquet(frames, index=False)

    model = _FakeModel()
    report = run_yoloe(
        frames,
        tmp_path / "objects",
        ObjectDetectionConfig(top_k=1, batch_size=1, device="cpu"),
        dataset_root=tmp_path,
        frame_store_id="fixture-v1",
        model=model,
    )

    raw = json.loads(
        (tmp_path / "objects/raw/v1/0000.json").read_text(encoding="utf-8")
    )
    detections = pd.read_parquet(tmp_path / "objects/detections.parquet")
    frames_artifact = pd.read_parquet(tmp_path / "objects/frames.parquet")

    assert raw["detection_class_entities"] == ["person"]
    assert raw["detection_scores"] == [0.9]
    assert detections[["label", "confidence"]].values.tolist() == [["person", 0.9]]
    assert frames_artifact[["frame_id", "detection_count"]].values.tolist() == [
        ["v1:0000", 1]
    ]
    assert report["source"] == "yoloe"
    assert report["completed_frames"] == 1
    assert report["failed_frames"] == 0
    assert model.calls[0]["conf"] == 0.2
    assert model.calls[0]["max_det"] == 1


def test_each_predict_call_sees_only_its_own_batch(tmp_path: Path) -> None:
    """Frames must not accumulate across iterations into one ever-growing call."""

    rows = []
    for index in range(4):
        image = tmp_path / f"keyframes/v1/{index:04d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture")
        rows.append(
            {
                "frame_id": f"v1:{index:04d}",
                "video_id": "v1",
                "frame_idx": index,
                "timestamp_ms": index * 100,
                "image_path": f"keyframes/v1/{index:04d}.jpg",
                "width": 20,
                "height": 10,
            }
        )
    frames = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(frames, index=False)

    model = _FakeModel()
    run_yoloe(
        frames,
        tmp_path / "objects",
        ObjectDetectionConfig(top_k=1, batch_size=2, device="cpu"),
        dataset_root=tmp_path,
        frame_store_id="fixture-v1",
        model=model,
    )

    assert [len(cast(list[str], call["images"])) for call in model.calls] == [2, 2]


def test_load_vocab_normalizes_deduplicates_and_preserves_order(
    tmp_path: Path,
) -> None:
    """Blank, duplicate, and differently cased entries collapse to one order."""

    path = tmp_path / "vocab.txt"
    path.write_text(
        "Road Sign\n\n  milestone  marker \nroad sign\ndam\n", encoding="utf-8"
    )

    assert load_vocab(path) == ["road sign", "milestone marker", "dam"]


@pytest.mark.parametrize("content", ["", "\n  \n"])
def test_load_vocab_rejects_an_empty_vocabulary(tmp_path: Path, content: str) -> None:
    """An empty prompt list must fail loudly instead of silencing detection."""

    path = tmp_path / "vocab.txt"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_vocab(path)


def test_run_yoloe_prompts_the_detector_with_the_configured_vocabulary(
    tmp_path: Path,
) -> None:
    """A configured vocabulary reaches the detector before any frame is scored."""

    image = tmp_path / "keyframes/v1/0000.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    frames = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": "v1:0000",
                "video_id": "v1",
                "frame_idx": 7,
                "timestamp_ms": 500,
                "image_path": "keyframes/v1/0000.jpg",
                "width": 20,
                "height": 10,
            }
        ]
    ).to_parquet(frames, index=False)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("person\ncar\ndam\n", encoding="utf-8")

    model = _FakeModel()
    run_yoloe(
        frames,
        tmp_path / "objects",
        ObjectDetectionConfig(batch_size=1, device="cpu", vocab_path=str(vocab)),
        dataset_root=tmp_path,
        model=model,
    )

    assert model.prompted == ["person", "car", "dam"]
