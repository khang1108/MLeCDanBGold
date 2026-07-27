from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image
from hcmai.enrichment.caption import CaptionConfig, FrameCaptioner, _atomic, generate_captions
class Backend:
    instances = 0
    def __init__(self, reject_red: bool = False):
        type(self).instances += 1; self.reject_red, self.calls = reject_red, []
    def __call__(self, images):
        self.calls.append(len(images))
        return [ValueError("rejected image") if self.reject_red and
                image.getpixel((0, 0)) == (255, 0, 0)
                else f"caption {image.getpixel((0, 0))}" for image in images]
def make_frames(root: Path, specs) -> Path:
    rows = []
    for index, spec in enumerate(specs):
        path = root / f"f{index}.png"
        if isinstance(spec, tuple): Image.new("RGB", (8, 8), spec).save(path)
        elif spec == "corrupt": path.write_bytes(b"not an image")
        rows.append({"frame_id": f"f{index}", "video_id": "v1", "frame_idx": index,
                     "timestamp_ms": index * 100, "image_path": path.name,
                     "width": 8, "height": 8})
    frames = root / "frames.parquet"; pd.DataFrame(rows).to_parquet(frames); return frames
def config() -> CaptionConfig:
    return CaptionConfig(model_checkpoint="fake/model", revision="fixture-revision",
                         batch_size=2, image_size=8,
                         write_interval=2, enrichment_version="caption_test", dataset_version="fixture_v1")
def test_batch_order_contract_black_and_completed_resume(tmp_path):
    Backend.instances = 0
    frames = make_frames(tmp_path, [(0, 0, 0), (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)])
    backend, cfg, output = Backend(), config(), tmp_path / "out"
    first = generate_captions(frames, output, cfg, FrameCaptioner(
        cfg, batch_fn=backend), dataset_root=tmp_path)
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    assert Backend.instances == 1 and backend.calls == [2, 2, 1]; assert table.frame_id.tolist() == [f"f{i}" for i in range(5)]
    assert table.caption.str.strip().ne("").all() and table.status.eq("completed").all()
    assert not table.duplicated(["frame_id", "enrichment_version"]).any()
    assert table.ocr_text.isna().all() and table.asr_text.isna().all(); assert table.detailed_caption.isna().all() and all(len(value) == 0 for value in table.objects)
    assert first["completed_count"] == 5 and first["failed_count"] == 0
    assert first["effective_configuration"]["batch_size"] == 2 and first["dataset_root"] == str(tmp_path)
    assert json.loads((output / "failures.json").read_text()) == []
    attempts, broken = [], FrameCaptioner(cfg)
    def fail():
        attempts.append(1)
        raise RuntimeError("unavailable")
    broken._load = fail
    with pytest.raises(RuntimeError):
        broken.caption_batch([Image.new("RGB", (1, 1))])
    results = broken.caption_batch([Image.new("RGB", (1, 1))])
    assert attempts == [1] and isinstance(results[0], RuntimeError)
    unused = Backend(); table["error_message"] = float("nan")
    table.to_parquet(output / "frame_enrichment.parquet", index=False)
    second = generate_captions(frames, output, cfg, FrameCaptioner(
        cfg, batch_fn=unused), dataset_root=tmp_path)
    assert unused.calls == [] and second["skipped_count"] == 5
    assert second["completed_count"] == 5 and second["retried_count"] == 0
    assert second["resolved_model_revision"] == "fixture-revision"
    for changed, name in ((replace(cfg, prompt="<DETAILED_CAPTION>"), "prompt"), (replace(cfg, model_checkpoint="other/model"), "model_checkpoint")):
        with pytest.raises(ValueError, match=name):
            generate_captions(frames, output, changed, dataset_root=tmp_path)
def test_resume_rejects_a_different_resolved_revision(tmp_path):
    frames, cfg, output = make_frames(tmp_path, [(0, 0, 0)]), config(), tmp_path / "out"
    first_captioner = FrameCaptioner(cfg, batch_fn=Backend())
    first_captioner.resolved_revision = "model-sha-one"
    first = generate_captions(frames, output, cfg, first_captioner, dataset_root=tmp_path)
    assert first["resolved_model_revision"] == "model-sha-one"
    second_captioner = FrameCaptioner(cfg, batch_fn=Backend())
    second_captioner.resolved_revision = "model-sha-two"
    with pytest.raises(ValueError, match="resolved_model_revision"):
        generate_captions(frames, output, cfg, second_captioner, dataset_root=tmp_path)
def test_explicit_failures_retry_and_malformed_row(tmp_path):
    frames = make_frames(tmp_path, [(255, 0, 0), (0, 255, 0), "missing",
                                    "corrupt", (0, 0, 0)])
    cfg, output, backend = config(), tmp_path / "out", Backend(reject_red=True)
    first = generate_captions(frames, output, cfg, FrameCaptioner(
        cfg, batch_fn=backend), dataset_root=tmp_path)
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    failures = json.loads((output / "failures.json").read_text())
    assert first["completed_count"] == 2 and first["failed_count"] == 3
    assert table.loc[table.frame_id == "f1", "status"].item() == "completed"
    assert table.loc[table.frame_id == "f4", "caption"].item() == "caption (0, 0, 0)"
    assert {item["processing_stage"] for item in failures} == {"image_load", "model"}
    assert all(item["error_message"] and item["exception_category"] for item in failures)
    for name, color in (("f2.png", (1, 1, 1)), ("f3.png", (2, 2, 2))):
        Image.new("RGB", (8, 8), color).save(tmp_path / name)
    retry = Backend()
    second = generate_captions(frames, output, cfg, FrameCaptioner(
        cfg, batch_fn=retry), dataset_root=tmp_path)
    assert second["skipped_count"] == 2 and second["retried_count"] == 3
    assert retry.calls == [2, 1] and second["completed_count"] == 5
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    table.loc[table.frame_id == "f0", "caption"] = ""
    table.to_parquet(output / "frame_enrichment.parquet", index=False)
    malformed = Backend()
    third = generate_captions(frames, output, cfg, FrameCaptioner(
        cfg, batch_fn=malformed), dataset_root=tmp_path)
    final = pd.read_parquet(output / "frame_enrichment.parquet")
    assert malformed.calls == [1] and third["retried_count"] == 1
    assert third["skipped_count"] == 4 and final.status.eq("completed").all(); assert not final.duplicated(["frame_id", "enrichment_version"]).any()
    target = output / "manifest.json"; before = target.read_bytes()
    def fail_write(path):
        path.write_bytes(b"partial"); raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        _atomic(target, fail_write)
    assert target.read_bytes() == before and not (output / "manifest.json.tmp").exists()
    (output / "frame_enrichment.parquet").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Cannot resume corrupted Parquet"):
        generate_captions(frames, output, cfg, dataset_root=tmp_path)
