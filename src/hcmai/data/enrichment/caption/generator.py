"""Bộ sinh (Generator) Caption.

Quản lý toàn bộ quá trình chạy mô hình sinh mô tả ảnh (captioning) cho các frames.

Các tính năng chính:
1. Lấy dữ liệu: Lấy danh sách các frames cần xử lý hình ảnh dựa trên manifest đầu vào.
2. Điều khiển Batch (Batching): Gom nhóm nhiều frames (vd: batch=32) để xử lý trên GPU nhanh hơn.
3. Ghi nhận kết quả: Lưu văn bản mô tả thu được vào các artifacts với cơ chế chống mất dữ liệu."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pandas as pd

from hcmai.common.utils.io import atomic_write, read_json, write_json
from hcmai.common.config import AppConfig
from hcmai.data.enrichment.caption.artifacts import write_caption_artifacts
from hcmai.data.enrichment.caption.adapters.transformers import TransformersCaptionAdapter
from hcmai.data.enrichment.caption.config import (
    DEFAULT_ENRICHMENT_CONFIG,
    CaptionConfig,
    CaptionJobConfig,
    ENRICHMENT_VERSION,
)
from hcmai.data.enrichment.caption.report import build_manifest
from hcmai.data.enrichment.caption.resume import guard_resume, resume_rows
from hcmai.data.enrichment.caption.runner import run_batches
from hcmai.data.enrichment.caption.models.contracts import CaptionAdapter
from hcmai.llm.pipeline import LLMService, LLMServiceConfig

def generate_captions(
    frames_path: str | Path,
    output_dir: str | Path,
    config: CaptionConfig,
    captioner: CaptionAdapter | None = None,
    *,
    dataset_root: str | Path = ".",
    frame_store_id: str | None = None,
) -> dict[str, Any]:
    """Generate or resume one deterministic caption enrichment artifact."""
    started, began, frames_path = datetime.now(timezone.utc), perf_counter(), Path(frames_path)
    root = Path(dataset_root).expanduser().resolve()

    frames = cast(
        list[dict[str, Any]],
        pd.read_parquet(frames_path).to_dict(orient="records"),
    )
    order = [str(frame["frame_id"]) for frame in frames]

    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    captioner = captioner or TransformersCaptionAdapter(config)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "manifest.json"
    captions_path = output / "captions.parquet"
    old = read_json(manifest_path) if manifest_path.exists() else {}

    # Resume only from source caption evidence, never from the legacy view.
    guard_resume(captions_path, old, config, root, frame_store_id=frame_store_id)
    rows, todo, skipped, retried = resume_rows(
        frames, captions_path, config, frame_store_id
    )
    resolved_revision = captioner.resolve_revision()
    guard_resume(
        captions_path,
        old,
        config,
        root,
        resolved_revision,
        frame_store_id=frame_store_id,
    )

    # Cập nhật thông tin vào manifest tạm thời
    provisional = {
        **old,
        ENRICHMENT_VERSION: config.enrichment_version,
        "effective_configuration": asdict(config),
        "dataset_root": str(root),
        "resolved_model_revision": resolved_revision,
        "frame_store_id": frame_store_id,
    }
    atomic_write(manifest_path, lambda path: write_json(provisional, path))

    # Xử lý từng batch
    failures: dict[str, dict[str, str]] = {}
    latencies = run_batches(
        todo,
        order,
        rows,
        failures,
        captioner,
        config,
        output,
        root,
        frame_store_id=frame_store_id,
        resolved_revision=resolved_revision,
    )

    if set(rows) != set(order):
        raise RuntimeError("caption processing did not retain one row per frame")

    write_caption_artifacts(output, order, rows, failures)
    manifest = build_manifest(
        config,
        frames_path,
        root,
        rows,
        captioner,
        started,
        perf_counter() - began,
        latencies,
        skipped,
        retried,
        frame_store_id=frame_store_id,
    )
    atomic_write(manifest_path, lambda path: write_json(manifest, path))
    return manifest

def main() -> int:
    """Run caption enrichment from YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_ENRICHMENT_CONFIG)
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    parser.add_argument("--model-config", default="llm/config.yaml")
    parser.add_argument("--frames")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    job = CaptionJobConfig.from_yaml(args.config)
    settings = AppConfig.from_yaml(args.app_config)
    captioner = None
    if settings.inference.enabled:
        from hcmai.data.enrichment.caption.adapters.remote import RemoteCaptionAdapter
        models = LLMServiceConfig.from_yaml(args.model_config)
        hosted = models.caption_generation
        expected = {
            "model_checkpoint": job.caption.model_checkpoint,
            "revision": job.caption.revision,
            "prompt": job.caption.prompt,
            "decoding": job.caption.decoding,
        }
        actual = {key: getattr(hosted, key) for key in expected}
        if actual != expected:
            raise ValueError(
                "Caption generation settings differ between enrichment and "
                "llm model configuration"
            )
        client = LLMService.remote(
            os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
            settings.inference.timeout_seconds,
        )
        captioner = RemoteCaptionAdapter(client, job.caption)
    manifest = generate_captions(
        args.frames or job.frames_path,
        args.output or job.output_dir,
        job.caption,
        captioner,
        dataset_root=args.dataset_root or job.dataset_root,
    )
    keys = "completed_count", "failed_count", "skipped_count", "retried_count"
    print({key: manifest[key] for key in keys})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
