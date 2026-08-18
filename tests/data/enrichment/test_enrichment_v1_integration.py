"""Exercise the complete offline HCMAI Enrichment V1 artifact chain.

The fixture uses injected adapters only. It verifies specialist failure
containment, deterministic context assembly, and context-only invalidation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from hcmai.data.enrichment.caption import generator as caption_generator
from hcmai.data.enrichment.ocr.models.entities import OCRRegionResult, OCRResult
from hcmai.data.enrichment.pipeline import EnrichmentJobConfig, EnrichmentService


class _CaptionAdapter:
    """Return one caption and one contained per-frame failure."""

    resolved_revision = "fixture-caption-revision"

    def resolve_revision(self) -> str:
        return self.resolved_revision

    def caption_batch(self, images: object) -> list[object]:
        del images
        return ["A person reads a sign.", RuntimeError("caption failed")]


class _OCRAdapter:
    """Return one usable OCR region and one empty OCR result."""

    resolved_revision = "fixture-ocr-revision"

    def recognize_batch(self, images: object) -> list[OCRResult]:
        del images
        return [
            OCRResult(
                text="HCMAI 2026",
                regions=(
                    OCRRegionResult(
                        text="HCMAI 2026",
                        confidence=0.9,
                        x_min=0.1,
                        y_min=0.1,
                        x_max=0.8,
                        y_max=0.3,
                    ),
                ),
            ),
            OCRResult(text="", regions=()),
        ]


def _write_fixture(root: Path) -> Path:
    """Write two canonical frames and repeated BTC object detections."""

    images = root / "images"
    images.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index in range(2):
        image = images / f"{index:04d}.jpg"
        Image.new("RGB", (16, 12), color=(index * 20, 0, 0)).save(image)
        rows.append(
            {
                "frame_id": f"f{index + 1}",
                "video_id": "v1",
                "frame_idx": index + 10,
                "timestamp_ms": index * 1000,
                "image_path": str(image),
                "width": 16,
                "height": 12,
            }
        )
    frames = root / "frame_store" / "frames.parquet"
    frames.parent.mkdir()
    pd.DataFrame(rows).to_parquet(frames, index=False)
    (frames.parent / "manifest.json").write_text(
        json.dumps({"frame_store_id": "btc-fixture-v1"}), encoding="utf-8"
    )

    for index in range(2):
        path = root / "btc_objects" / "v1" / f"{index:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "detection_class_entities": ["person", "person"],
                    "detection_scores": [0.9, 0.8],
                    "detection_boxes": [
                        [0.0, 0.0, 0.5, 0.5],
                        [0.2, 0.2, 0.9, 0.9],
                    ],
                }
            ),
            encoding="utf-8",
        )
    return frames


def _write_config(root: Path, frames: Path) -> Path:
    """Write the exact config sections consumed by every offline stage."""

    path = root / "enrichment.yaml"
    path.write_text(
        f"""
dataset:
  version: fixture-v1
  source: btc_keyframes
  btc_root: {root}
  data_root: {root}
  frame_store_id: btc-fixture-v1
  frames_path: {frames}
  frame_store_output: {frames.parent}
  objects_root: {root / 'btc_objects'}
caption:
  output_dir: {root / 'captions'}
  model_checkpoint: fixture/caption
  revision: fixture-caption-revision
  prompt: <CAPTION>
  decoding: {{}}
  device: cpu
  precision: fp32
  dtype: float32
  image_size: 16
  batch_size: 2
  enrichment_version: caption-v1
  write_interval: 2
ocr:
  output_dir: {root / 'ocr'}
  enabled: true
  backend: fixture
  checkpoint: fixture/ocr
  revision: fixture-ocr-revision
  device: cpu
  dtype: float32
  batch_size: 2
  image_size: 16
  enrichment_version: ocr-enrichment-v1
  artifact_version: ocr-v1
  min_region_confidence: 0.5
  min_context_quality: 0.5
objects:
  output_dir: {root / 'objects'}
  artifact_version: object-v1
  summary_min_confidence: 0.25
  max_summary_labels: 20
context:
  output_dir: {root / 'context'}
  context_version: frame-context-v1
  caption_token_budget: 80
  ocr_token_budget: 80
  object_token_budget: 40
  min_ocr_quality: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_two_frame_offline_enrichment_and_context_only_rebuild(
    tmp_path: Path, monkeypatch
) -> None:
    """Generate specialists once, then invalidate only derived context policy."""

    frames = _write_fixture(tmp_path)
    job = EnrichmentJobConfig.from_yaml(_write_config(tmp_path, frames))

    caption_report = EnrichmentService.generate_captions(
        job.frames_path,
        job.caption_output_dir,
        job.caption,
        _CaptionAdapter(),
        dataset_root=job.data_root,
        frame_store_id=job.frame_store_id,
    )
    ocr_report = EnrichmentService.generate_ocr(
        job.frames_path,
        job.ocr_output_dir,
        job.ocr,
        _OCRAdapter(),
        dataset_root=job.data_root,
        frame_store_id=job.frame_store_id,
    )
    object_report = EnrichmentService.import_objects(
        job.frames_path,
        job.objects_root,
        job.object_output_dir,
        job.objects,
        frame_store_id=job.frame_store_id,
    )
    context_path = EnrichmentService.build_frame_context_from_job(job)

    assert caption_report["completed_count"] == 1
    assert caption_report["failed_count"] == 1
    assert ocr_report["completed_frames"] == 2
    assert object_report["detection_count"] == 4
    rows = pd.read_parquet(context_path).set_index("frame_id")
    assert "[CAPTION]\nA person reads a sign." in rows.loc["f1", "context_text"]
    assert "[VISIBLE_TEXT]\nHCMAI 2026" in rows.loc["f1", "context_text"]
    assert "[OBJECTS]\nperson x2" in rows.loc["f1", "context_text"]
    assert "[CAPTION]" not in rows.loc["f2", "context_text"]
    assert "None" not in rows.loc["f2", "context_text"]

    monkeypatch.setattr(
        EnrichmentService,
        "generate_captions",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("context rebuild generated captions")
        ),
    )
    monkeypatch.setattr(
        EnrichmentService,
        "generate_ocr",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("context rebuild generated OCR")
        ),
    )
    monkeypatch.setattr(
        EnrichmentService,
        "import_objects",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("context rebuild imported objects")
        ),
    )

    changed = replace(job.context, object_token_budget=1)
    rebuilt = EnrichmentService.build_frame_context_from_job(
        replace(job, context=changed)
    )
    manifest = json.loads((rebuilt.parent / "manifest.json").read_text())
    assert manifest["serializer_config"]["object_token_budget"] == 1


def test_checked_in_enrichment_config_loads_all_v1_stage_contracts() -> None:
    """Keep checked-in paths and specialist policies accepted by one loader."""

    job = EnrichmentJobConfig.from_yaml("configs/enrichment.yaml")

    assert job.source == "btc_keyframes"
    assert job.caption_output_dir == Path("artifacts/enrichment/captions")
    assert job.ocr_output_dir == Path("artifacts/enrichment/ocr")
    assert job.object_output_dir == Path("artifacts/enrichment/objects")
    assert job.transcript_output_dir == Path("artifacts/enrichment/transcripts")
    assert job.context_output_dir == Path("artifacts/enrichment/context")
    assert job.context == replace(job.context)
    assert (
        job.context.caption_token_budget,
        job.context.ocr_token_budget,
        job.context.object_token_budget,
    ) == (80, 80, 40)


def test_caption_command_propagates_configured_frame_store_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep independently run captions compatible with the context join."""

    frames = _write_fixture(tmp_path)
    config_path = _write_config(tmp_path, frames)
    captured: dict[str, object] = {}

    def fake_generate_captions(*args: object, **kwargs: object) -> dict[str, int]:
        """Capture the command's public generation call without model inference."""

        captured["frame_store_id"] = kwargs.get("frame_store_id")
        return {
            "completed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "retried_count": 0,
        }

    monkeypatch.setattr(caption_generator, "generate_captions", fake_generate_captions)
    monkeypatch.setattr(
        caption_generator.AppConfig,
        "from_yaml",
        lambda path: SimpleNamespace(inference=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_enrichment.py", "--config", str(config_path)],
    )

    assert caption_generator.main() == 0
    assert captured["frame_store_id"] == "btc-fixture-v1"
