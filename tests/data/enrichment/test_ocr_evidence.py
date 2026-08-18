"""Behavior tests for lossless, region-aware OCR enrichment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest
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

    def __init__(
        self, outputs: list[OCRResult], *, resolved_revision: str = "r1"
    ) -> None:
        self.outputs = outputs
        self.resolved_revision = resolved_revision
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


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf")])
def test_florence_rejects_non_finite_quad_coordinates(coordinate):
    with pytest.raises(ValueError, match="finite"):
        _parse_regions(
            {
                "labels": ["bad"],
                "quad_boxes": [[coordinate, 0, 1, 0, 1, 1, 0, 1]],
            },
            image_size=(10, 10),
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


def test_requested_revision_change_reprocesses_all_rows(tmp_path):
    source = _frames(tmp_path)
    outputs = [
        OCRResult("one", (_region("one"),)),
        OCRResult("two", (_region("two"),)),
    ]
    generate_ocr(source, tmp_path / "ocr", _config(), engine=_Engine(outputs))

    changed = replace(_config(), revision="r2")
    retry = _Engine(outputs, resolved_revision="r2")
    report = generate_ocr(source, tmp_path / "ocr", changed, engine=retry)

    rows = pd.read_parquet(tmp_path / "ocr/frames.parquet")
    assert retry.calls == [2]
    assert report["skipped_frames"] == 0
    assert report["retried_frames"] == 2
    assert rows.model_revision.tolist() == ["r2", "r2"]


def test_partial_retry_runtime_revision_change_reprocesses_reused_rows(tmp_path):
    source = _frames(tmp_path)
    outputs = [
        OCRResult("one", (_region("one"),)),
        OCRResult("two", (_region("two"),)),
    ]
    generate_ocr(source, tmp_path / "ocr", _config(), engine=_Engine(outputs))
    regions_path = tmp_path / "ocr/regions.parquet"
    regions = pd.read_parquet(regions_path)
    regions[regions.frame_id != "f1"].to_parquet(regions_path, index=False)

    retry = _Engine(outputs, resolved_revision="r2")
    report = generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine_factory=lambda _: retry,
    )

    rows = pd.read_parquet(tmp_path / "ocr/frames.parquet")
    assert retry.calls == [1, 2]
    assert report["processed_frames"] == 2
    assert report["skipped_frames"] == 0
    assert report["retried_frames"] == 2
    assert rows.model_revision.tolist() == ["r2", "r2"]


def test_missing_canonical_frames_does_not_overwrite_existing_artifacts(tmp_path):
    source = _frames(tmp_path)
    outputs = [
        OCRResult("one", (_region("one"),)),
        OCRResult("two", (_region("two"),)),
    ]
    output = tmp_path / "ocr"
    generate_ocr(source, output, _config(), engine=_Engine(outputs))
    before = {
        name: (output / name).read_bytes()
        for name in ("frames.parquet", "regions.parquet", "manifest.json")
    }
    source.unlink()

    with pytest.raises(FileNotFoundError):
        generate_ocr(source, output, _config(), engine=_Engine(outputs))

    assert {
        name: (output / name).read_bytes()
        for name in before
    } == before


def test_generation_sanitizes_raw_output_before_publishing_complete_set(tmp_path):
    source = _frames(tmp_path, count=1)
    raw = {
        "array": np.asarray([1, 2], dtype=np.int64),
        "nested": [float("nan"), {"infinite": float("inf")}],
        "unsupported": object(),
    }

    generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine=_Engine([OCRResult("one", (_region("one"),), raw)]),
    )

    output = tmp_path / "ocr"
    assert {
        "frames.parquet",
        "regions.parquet",
        "failures.json",
        "manifest.json",
        "frame_enrichment.parquet",
        "ocr_report.json",
    } <= {path.name for path in output.iterdir()}
    report = json.loads((output / "ocr_report.json").read_text())
    assert report["raw_evidence"][0]["raw_output"] == {
        "array": [1, 2],
        "nested": [None, {"infinite": None}],
        "unsupported": None,
    }


def test_revision_full_reprocess_counts_only_prior_rows_as_retried(tmp_path):
    source = _frames(tmp_path, count=1)
    generate_ocr(
        source,
        tmp_path / "ocr",
        _config(),
        engine=_Engine([OCRResult("one", (_region("one"),))]),
    )

    expanded = _frames(tmp_path, count=2)
    outputs = [
        OCRResult("one", (_region("one"),)),
        OCRResult("two", (_region("two"),)),
    ]
    retry = _Engine(outputs, resolved_revision="r2")
    report = generate_ocr(
        expanded,
        tmp_path / "ocr",
        _config(),
        engine_factory=lambda _: retry,
    )

    assert retry.calls == [1, 2]
    assert report["processed_frames"] == 2
    assert report["retried_frames"] == 1
