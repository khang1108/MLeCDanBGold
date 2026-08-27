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

    resolved_revision: str | None = "fixture-caption-revision"

    def resolve_revision(self) -> str:
        assert self.resolved_revision is not None
        return self.resolved_revision

    def caption_batch(self, images: object) -> list[object]:
        del images
        return ["A person reads a sign.", RuntimeError("caption failed")]


class _OCRAdapter:
    """Return one usable OCR region and one empty OCR result."""

    resolved_revision: str | None = "fixture-ocr-revision"

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
  model: fixture/yoloe
  min_confidence: 0.2
  top_k: 30
  batch_size: 2
  device: cpu
  artifact_version: object-yoloe-v1
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
    object_report = EnrichmentService.detect_objects(
        job.frames_path,
        job.object_output_dir,
        job.objects,
        raw_output_root=tmp_path / "btc_objects",
        frame_store_id=job.frame_store_id,
    )
    context_path = EnrichmentService.build_frame_context_from_job(job)

    assert caption_report["completed_count"] == 1
    assert caption_report["failed_count"] == 1
    assert ocr_report["completed_frames"] == 2
    assert object_report["detection_count"] == 4
    rows = pd.read_parquet(context_path).set_index("frame_id")
    for artifact in (
        job.caption_output_dir / "captions.parquet",
        job.ocr_output_dir / "frames.parquet",
        job.object_output_dir / "frames.parquet",
        context_path,
    ):
        identity = pd.read_parquet(artifact).set_index("frame_id")
        assert identity.loc["f1", ["video_id", "frame_idx", "timestamp_ms"]].tolist() == [
            "v1",
            10,
            0,
        ]
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
        "detect_objects",
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
    project_root = Path(__file__).resolve().parents[3]

    assert job.source == "btc_keyframes"
    assert job.caption_output_dir == project_root / "artifacts/enrichment/captions"
    assert job.ocr_output_dir == project_root / "artifacts/enrichment/ocr"
    assert job.object_output_dir == project_root / "artifacts/enrichment/objects"
    assert job.transcript_output_dir == project_root / "artifacts/enrichment/transcripts"
    assert job.context_output_dir == project_root / "artifacts/enrichment/context"
    assert job.context == replace(job.context)
    assert (
        job.context.caption_token_budget,
        job.context.ocr_token_budget,
        job.context.object_token_budget,
    ) == (80, 80, 40)


def test_custom_job_paths_resolve_from_project_root_not_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """Keep a custom config's relative artifact paths independent of the shell."""

    config_path = tmp_path / "enrichment.yaml"
    config_path.write_text(
        """
dataset:
  version: fixture-v1
  source: btc_keyframes
  btc_root: fixture/btc
  data_root: fixture/data
  frame_store_id: fixture-frames-v1
  frames_path: fixture/artifacts/frame_store/frames.parquet
  frame_store_output: fixture/artifacts/frame_store
caption:
  output_dir: fixture/artifacts/captions
  model_checkpoint: fixture/caption
  revision: fixture-caption-revision
  prompt: <CAPTION>
  decoding: {}
  device: cpu
  precision: fp32
  dtype: float32
  image_size: 16
  batch_size: 1
  enrichment_version: caption-v1
  write_interval: 1
ocr:
  output_dir: fixture/artifacts/ocr
  enabled: true
  backend: fixture
  checkpoint: fixture/ocr
  revision: fixture-ocr-revision
  device: cpu
  dtype: float32
  batch_size: 1
  image_size: 16
  enrichment_version: ocr-enrichment-v1
  artifact_version: ocr-v1
  min_region_confidence: 0.0
  min_context_quality: 0.5
objects:
  output_dir: fixture/artifacts/objects
  model: fixture/yoloe
  min_confidence: 0.2
  top_k: 30
  batch_size: 1
  device: cpu
  artifact_version: object-yoloe-v1
  summary_min_confidence: 0.25
  max_summary_labels: 20
transcript:
  output_dir: fixture/artifacts/transcripts
context:
  output_dir: fixture/artifacts/context
  context_version: frame-context-v1
  caption_token_budget: 80
  ocr_token_budget: 80
  object_token_budget: 40
  min_ocr_quality: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    job = EnrichmentJobConfig.from_yaml(config_path)
    project_root = Path(__file__).resolve().parents[3]

    assert job.frames_path == project_root / "fixture/artifacts/frame_store/frames.parquet"
    assert job.caption_output_dir == project_root / "fixture/artifacts/captions"
    assert job.context_output_dir == project_root / "fixture/artifacts/context"


def test_default_operations_keep_btc_specialist_lineage_for_context(
    tmp_path: Path, monkeypatch
) -> None:
    """Run BTC specialists through the active operations before context assembly."""

    from hcmai.data.corpus_build.pipeline import DefaultPreparationOperations
    from hcmai.data.enrichment.object_detection import materialize_object_artifacts

    frames = _write_fixture(tmp_path)
    job = EnrichmentJobConfig.from_yaml(_write_config(tmp_path, frames))
    operations = object.__new__(DefaultPreparationOperations)
    object.__setattr__(operations, "config", SimpleNamespace(
        corpus_revision=job.dataset_version,
        frame_store_source="btc_keyframes",
        preprocessing=SimpleNamespace(device="cpu"),
        models=SimpleNamespace(
            ocr=SimpleNamespace(
                model_name=job.ocr.checkpoint,
                revision=job.ocr.revision,
            )
        ),
    ))
    object.__setattr__(operations, "paths", SimpleNamespace(
        frames_path=job.frames_path,
        frame_store_root=job.frame_store_output,
        caption_root=job.caption_output_dir,
        ocr_root=job.ocr_output_dir,
        object_root=job.object_output_dir,
        context_root=job.context_output_dir,
        asr_enrichment_path=tmp_path / "asr" / "frame_enrichment.parquet",
        transcripts_root=tmp_path / "transcripts",
    ))
    operations.enrichment_job = job
    object.__setattr__(
        operations, "caption_job", SimpleNamespace(caption=job.caption)
    )
    object.__setattr__(
        operations, "_current_run_id", "s3-run-id-must-not-be-frame-lineage"
    )
    operations._remote_pool = lambda capability: None

    monkeypatch.setattr(
        EnrichmentService,
        "create_caption_adapter",
        lambda config: _CaptionAdapter(),
    )
    monkeypatch.setattr(
        EnrichmentService,
        "create_ocr_adapter",
        lambda config: _OCRAdapter(),
    )
    monkeypatch.setattr(
        EnrichmentService,
        "detect_objects",
        lambda frames_path, output_dir, config, **kwargs: materialize_object_artifacts(
            frames_path,
            tmp_path / "btc_objects",
            output_dir,
            config,
            frame_store_id=kwargs["frame_store_id"],
        ),
    )

    operations.generate_caption()
    operations.generate_ocr()
    operations.detect_objects()
    context_path = operations.build_frame_context()

    assert context_path.exists()
    assert set(pd.read_parquet(context_path)["frame_id"]) == {"f1", "f2"}
    for artifact in (
        job.caption_output_dir / "captions.parquet",
        job.ocr_output_dir / "frames.parquet",
        job.object_output_dir / "frames.parquet",
    ):
        assert set(pd.read_parquet(artifact)["frame_store_id"]) == {
            job.frame_store_id
        }


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
