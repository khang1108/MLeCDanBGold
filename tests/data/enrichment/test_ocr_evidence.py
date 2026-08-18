"""Behavior tests for lossless, region-aware OCR enrichment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from hcmai.common.schemas import OCRItem, OCRResponse
from hcmai.common.schemas.inference import OCRRegionItem
from hcmai.data.enrichment.ocr.adapters.remote import RemoteOCRAdapter
from hcmai.data.enrichment.ocr.adapters.florence import _parse_regions
from hcmai.data.enrichment.ocr.artifacts import normalize_regions
from hcmai.data.enrichment.ocr.config import OCRConfig
from hcmai.data.enrichment.ocr.generator import generate_ocr
from hcmai.data.enrichment.ocr.models.entities import OCRRegionResult, OCRResult


class _Engine:
    """Return fixture OCR responses and record batch sizes."""

    resolved_revision = "r1"

    def __init__(self, outputs: list[OCRResult]) -> None:
        self.outputs = outputs
        self.calls: list[int] = []

    def recognize_batch(self, images: list[Image.Image]) -> list[OCRResult]:
        self.calls.append(len(images))
        return self.outputs[: len(images)]


class _RemoteClient:
    def readiness(self):
        raise AssertionError("readiness is not needed after revision is pinned")

    def ocr(self, images):
        return OCRResponse(
            model="fake/model",
            revision="r1",
            items=[
                OCRItem(
                    item_id="0",
                    text="AIC 2026",
                    raw_output={"labels": ["AIC 2026"]},
                    regions=[
                        OCRRegionItem(
                            text="AIC 2026",
                            confidence=None,
                            x_min=0.1,
                            y_min=0.2,
                            x_max=0.3,
                            y_max=0.4,
                        )
                    ],
                )
            ],
            latency_ms=1,
        )


def _config() -> OCRConfig:
    return OCRConfig(
        checkpoint="fake/model",
        revision="r1",
        batch_size=2,
        image_size=None,
        dataset_version="fixture",
    )


def _frames(tmp_path: Path, count: int = 2) -> Path:
    rows = []
    for index in range(count):
        image_path = tmp_path / f"f{index}.png"
        Image.new("RGB", (10, 20), "white").save(image_path)
        rows.append(
            {
                "frame_id": f"f{index}",
                "video_id": "v1",
                "frame_idx": index,
                "timestamp_ms": index * 100,
                "image_path": str(image_path),
                "width": 10,
                "height": 20,
            }
        )
    path = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _region(text: str, confidence: float | None = None) -> OCRRegionResult:
    return OCRRegionResult(text, confidence, 0.1, 0.2, 0.3, 0.4)


def test_normalization_preserves_raw_regions_and_filters_only_derived_text():
    regions = (
        _region("  Cafe\u0301   Việt  ", 0.9),
        _region("||||", 0.8),
        _region("CAFÉ Việt", 0.7),
        _region("low confidence", 0.2),
    )

    normalized = normalize_regions(regions, min_confidence=0.5)

    assert normalized.text == "Café Việt"
    assert normalized.usable_region_count == 1
    assert normalized.quality_score == 0.1625
    assert regions[0].text == "  Cafe\u0301   Việt  "
    assert len(regions) == 4


def test_remote_ocr_round_trip_preserves_regions_exactly():
    adapter = RemoteOCRAdapter(_RemoteClient(), _config())
    adapter.resolved_revision = "r1"

    result = adapter.recognize_batch([Image.new("RGB", (1, 1))])[0]

    assert result == OCRResult(
        text="AIC 2026",
        regions=(_region("AIC 2026"),),
        raw_output={"labels": ["AIC 2026"]},
    )


def test_florence_region_quadrilaterals_are_axis_aligned_and_clamped():
    regions = _parse_regions(
        {
            "labels": ["first", "second"],
            "quad_boxes": [
                [-5, 4, 5, 2, 12, 8, 1, 9],
                [2, 6, 4, 6, 4, 10, 2, 10],
            ],
        },
        image_size=(10, 8),
    )

    assert regions == (
        OCRRegionResult("first", None, 0.0, 0.25, 1.0, 1.0),
        OCRRegionResult("second", None, 0.2, 0.75, 0.4, 1.0),
    )


def test_parquet_rows_and_regions_keep_canonical_input_order(tmp_path):
    source = _frames(tmp_path)
    outputs = [
        OCRResult("second\nfirst", (_region("second"), _region("first"))),
        OCRResult("only", (_region("only"),)),
    ]

    report = generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine=_Engine(outputs),
        frame_store_id="btc-v1",
    )

    frames = pd.read_parquet(tmp_path / "ocr/frames.parquet")
    regions = pd.read_parquet(tmp_path / "ocr/regions.parquet")
    projection = pd.read_parquet(tmp_path / "ocr/frame_enrichment.parquet")
    assert frames.frame_id.tolist() == ["f0", "f1"]
    assert frames.frame_idx.tolist() == [0, 1]
    assert frames.raw_text.tolist() == ["second\nfirst", "only"]
    assert regions[["frame_id", "region_order", "text"]].values.tolist() == [
        ["f0", 0, "second"],
        ["f0", 1, "first"],
        ["f1", 0, "only"],
    ]
    assert projection.frame_id.tolist() == ["f0", "f1"]
    assert report["frames_with_raw_text"] == 2
    assert report["frames_with_normalized_text"] == 2
    assert report["frames_with_regions"] == 2
    assert report["raw_region_count"] == 3
    assert report["usable_region_count"] == 3
    assert report["mean_quality_score"] == 1.0
    assert "accuracy" not in " ".join(report).casefold()


def test_resume_retries_failed_and_inconsistent_region_rows(tmp_path):
    source = _frames(tmp_path)
    first = _Engine(
        [
            OCRResult("one", (_region("one"),)),
            OCRResult("two", (_region("two"),)),
        ]
    )
    generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine=first,
        frame_store_id="btc-v1",
    )

    regions_path = tmp_path / "ocr/regions.parquet"
    regions = pd.read_parquet(regions_path)
    regions = regions[regions.frame_id != "f1"]
    regions.to_parquet(regions_path, index=False)

    retry = _Engine([OCRResult("two fixed", (_region("two fixed"),))])
    report = generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine=retry,
        frame_store_id="btc-v1",
    )

    frames = pd.read_parquet(tmp_path / "ocr/frames.parquet")
    regions = pd.read_parquet(regions_path)
    assert retry.calls == [1]
    assert report["skipped_frames"] == 1
    assert report["retried_frames"] == 1
    assert frames.normalized_text.tolist() == ["one", "two fixed"]
    assert regions.text.tolist() == ["one", "two fixed"]
