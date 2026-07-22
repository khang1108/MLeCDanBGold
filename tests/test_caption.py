from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from hcmai.caption import CaptionConfig, FrameCaptioner, generate_captions
class Backend:
    instances = 0

    def __init__(self, reject_red: bool = False):
        type(self).instances += 1
        self.reject_red, self.calls = reject_red, []
    def __call__(self, images):
        self.calls.append(len(images))
        return [ValueError("rejected image") if self.reject_red
                and image.getpixel((0, 0)) == (255, 0, 0)
                else f"caption {image.getpixel((0, 0))}" for image in images]
def make_frames(root: Path, specs) -> Path:
    rows = []
    for index, spec in enumerate(specs):
        path = root / f"f{index}.png"
        if isinstance(spec, tuple):
            Image.new("RGB", (8, 8), spec).save(path)
        elif spec == "corrupt":
            path.write_bytes(b"not an image")
        rows.append({"frame_id": f"f{index}", "video_id": "v1",
                     "frame_idx": index, "timestamp_ms": index * 100,
                     "image_path": str(path), "width": 8, "height": 8})
    frames = root / "frames.parquet"
    pd.DataFrame(rows).to_parquet(frames)
    return frames
def config() -> CaptionConfig:
    return CaptionConfig(model_checkpoint="fake/model", batch_size=2,
                         image_size=8, write_interval=2,
                         enrichment_version="caption_test",
                         dataset_version="fixture_v1")
def test_batch_order_contract_black_and_completed_resume(tmp_path):
    Backend.instances = 0
    frames = make_frames(tmp_path, [(0, 0, 0), (1, 2, 3), (4, 5, 6),
                                    (7, 8, 9), (10, 11, 12)])
    backend, cfg, output = Backend(), config(), tmp_path / "out"
    first = generate_captions(
        frames, output, cfg, FrameCaptioner(cfg, batch_fn=backend))
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    assert Backend.instances == 1 and backend.calls == [2, 2, 1]
    assert table.frame_id.tolist() == [f"f{i}" for i in range(5)]
    assert table.caption.str.strip().ne("").all() and table.status.eq("completed").all()
    assert not table.duplicated(["frame_id", "enrichment_version"]).any()
    assert table.ocr_text.isna().all() and table.asr_text.isna().all()
    assert table.detailed_caption.isna().all() and all(len(value) == 0 for value in table.objects)
    assert first["completed_count"] == 5 and first["failed_count"] == 0
    assert first["effective_configuration"]["batch_size"] == 2
    assert json.loads((output / "failures.json").read_text()) == []
    attempts, broken = [], FrameCaptioner(cfg)
    def fail():
        attempts.append(1)
        raise RuntimeError("unavailable")
    broken._load = fail
    try:
        broken.caption_batch([Image.new("RGB", (1, 1))])
    except RuntimeError:
        pass
    results = broken.caption_batch([Image.new("RGB", (1, 1))])
    assert attempts == [1] and isinstance(results[0], RuntimeError)
    unused = Backend()
    second = generate_captions(
        frames, output, cfg, FrameCaptioner(cfg, batch_fn=unused))
    assert unused.calls == [] and second["skipped_count"] == 5
    assert second["completed_count"] == 5 and second["retried_count"] == 0
def test_explicit_failures_retry_and_malformed_row(tmp_path):
    frames = make_frames(tmp_path, [(255, 0, 0), (0, 255, 0), "missing",
                                    "corrupt", (0, 0, 0)])
    cfg, output, backend = config(), tmp_path / "out", Backend(reject_red=True)
    first = generate_captions(
        frames, output, cfg, FrameCaptioner(cfg, batch_fn=backend))
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    failures = json.loads((output / "failures.json").read_text())
    assert first["completed_count"] == 2 and first["failed_count"] == 3
    assert table.loc[table.frame_id == "f1", "status"].item() == "completed"
    assert table.loc[table.frame_id == "f4", "caption"].item() == "caption (0, 0, 0)"
    assert {item["processing_stage"] for item in failures} == {"image_load", "model"}
    assert all(item["error_message"] and item["exception_category"] for item in failures)
    Image.new("RGB", (8, 8), (1, 1, 1)).save(tmp_path / "f2.png")
    Image.new("RGB", (8, 8), (2, 2, 2)).save(tmp_path / "f3.png")
    retry = Backend()
    second = generate_captions(
        frames, output, cfg, FrameCaptioner(cfg, batch_fn=retry))
    assert second["skipped_count"] == 2 and second["retried_count"] == 3
    assert retry.calls == [2, 1] and second["completed_count"] == 5
    table = pd.read_parquet(output / "frame_enrichment.parquet")
    table.loc[table.frame_id == "f0", "caption"] = ""
    table.to_parquet(output / "frame_enrichment.parquet", index=False)
    malformed = Backend()
    third = generate_captions(
        frames, output, cfg, FrameCaptioner(cfg, batch_fn=malformed))
    final = pd.read_parquet(output / "frame_enrichment.parquet")
    assert malformed.calls == [1] and third["retried_count"] == 1
    assert third["skipped_count"] == 4 and final.status.eq("completed").all()
    assert not final.duplicated(["frame_id", "enrichment_version"]).any()
