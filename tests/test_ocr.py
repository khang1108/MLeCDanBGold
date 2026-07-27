from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image
from hcmai.common.schemas import FrameEnrichment
from hcmai.enrichment.ocr import OCRConfig, OCRResult, generate_ocr
class Engine:
    instances = 0
    def __init__(self, outputs=None, error=None):
        type(self).instances += 1
        self.outputs, self.error, self.calls = outputs, error, []
        self.resolved_revision = "fake-revision"
    def recognize_batch(self, images):
        self.calls.append(len(images))
        if self.error:
            raise self.error
        if self.outputs is not None:
            return self.outputs[:len(images)]
        return [OCRResult(
            "" if image.getpixel((0, 0))[0] == 0 else "Cafe\u0301 Việt\n  12:30",
            {"value": image.getpixel((0, 0))[0]}, .9) for image in images]
def frames(root: Path, specs) -> Path:
    rows = []
    for index, spec in enumerate(specs):
        path = root / f"f{index}.png"
        if isinstance(spec, int):
            Image.new("RGB", (8, 8), (spec, spec, spec)).save(path)
        elif spec == "corrupt":
            path.write_bytes(b"broken")
        rows.append({"frame_id": f"f{index}", "video_id": "v", "frame_idx": index, "timestamp_ms": index,
                     "image_path": str(path), "width": 8, "height": 8})
    result = root / "frames.parquet"
    pd.DataFrame(rows).to_parquet(result)
    return result
def config(**updates):
    values = {"checkpoint": "fake/model", "revision": "r1", "batch_size": 2,
              "image_size": 8, "enrichment_version": "ocr_test", "dataset_version": "fixture"}
    values.update(updates)
    return OCRConfig(**values)
def test_disabled_and_empty_do_not_construct_or_decode(tmp_path):
    source = frames(tmp_path, ["missing"])
    report = generate_ocr(
        source, tmp_path / "off", config(enabled=False),
        engine_factory=lambda _: pytest.fail("constructed"))
    table = pd.read_parquet(tmp_path / "off/frame_enrichment.parquet")
    assert table.empty and report["enabled"] is False
    assert report["total_frames"] == 1 and report["disabled_frames"] == 1
    empty = tmp_path / "empty.parquet"
    pd.DataFrame(columns=pd.read_parquet(source).columns).to_parquet(empty)
    assert generate_ocr(empty, tmp_path / "empty", config())["total_frames"] == 0
def test_batch_normalization_contract_report_and_resume(tmp_path):
    Engine.instances = 0
    source, made = frames(tmp_path, [0, 1, 2, 3, 4]), []
    report = generate_ocr(
        source, tmp_path / "out", config(),
        engine_factory=lambda _: made.append(Engine()) or made[0])
    table = pd.read_parquet(tmp_path / "out/frame_enrichment.parquet")
    assert Engine.instances == 1 and made[0].calls == [2, 2, 1]
    assert table.frame_id.tolist() == [f"f{i}" for i in range(5)]
    assert pd.isna(table.loc[0, "ocr_text"])
    assert table.loc[1, "ocr_text"] == "Café Việt 12:30"
    assert table.caption.isna().all() and table.asr_text.isna().all()
    assert table.detailed_caption.isna().all() and table.status.eq("completed").all()
    assert not table.duplicated(["frame_id", "enrichment_version"]).any()
    assert report["frames_with_text"] == 4 and report["empty_text_frames"] == 1
    assert report["confidence_available"] and report["raw_output_available"]
    assert len(report["raw_evidence"]) == 5
    assert all(FrameEnrichment.model_validate({key: (None if pd.isna(value) else value)
        for key, value in row.items() if key != "objects"} | {"objects": list(row["objects"])})
        for row in table.to_dict("records"))
    resumed = generate_ocr(
        source, tmp_path / "out", config(),
        engine_factory=lambda _: pytest.fail("loaded"))
    assert resumed["skipped_frames"] == 5 and resumed["processed_frames"] == 0
    assert json.loads((tmp_path / "out/failures.json").read_text()) == []
def test_relative_image_paths_resolve_against_dataset_root(tmp_path):
    source = frames(tmp_path, [1])
    table = pd.read_parquet(source)
    table["image_path"] = "f0.png"
    table.to_parquet(source, index=False)
    report = generate_ocr(
        source, tmp_path / "out", config(), engine=Engine(), dataset_root=tmp_path)
    assert report["completed_frames"] == 1
    assert report["dataset_root"] == str(tmp_path)
@pytest.mark.parametrize("outputs", [
    [OCRResult(3)], [OCRResult("text", confidence=float("nan"))], [],
])
def test_malformed_backend_results_are_explicit(tmp_path, outputs):
    source = frames(tmp_path, [1])
    report = generate_ocr(
        source, tmp_path / "out", config(), engine=Engine(outputs=outputs))
    table = pd.read_parquet(tmp_path / "out/frame_enrichment.parquet")
    assert report["failed_frames"] == 1 and table.status.item() == "failed"
    assert table.error_message.item() and len(table.error_message.item()) <= 300
def test_missing_corrupt_batch_failure_retry_and_black(tmp_path):
    source = frames(tmp_path, [1, "missing", "corrupt", 0, 2])
    first = generate_ocr(
        source, tmp_path / "out", config(), engine=Engine(error=TimeoutError("x" * 500)))
    assert first["failed_frames"] == 5
    Image.new("RGB", (8, 8), 3).save(tmp_path / "f1.png")
    Image.new("RGB", (8, 8), 4).save(tmp_path / "f2.png")
    engine = Engine()
    second = generate_ocr(source, tmp_path / "out", config(), engine=engine)
    table = pd.read_parquet(tmp_path / "out/frame_enrichment.parquet")
    assert second["retried_frames"] == 5 and second["completed_frames"] == 5
    assert engine.calls == [2, 2, 1] and table.loc[3, "status"] == "completed"
    assert pd.isna(table.loc[3, "ocr_text"])
