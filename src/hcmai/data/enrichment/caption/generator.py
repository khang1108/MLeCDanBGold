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
from typing import Any, Sequence

from hcmai.common.utils.io import read_json
from hcmai.common.config import AppConfig
from hcmai.data.enrichment.caption.artifacts import write_caption_artifacts
from hcmai.data.enrichment.caption.adapters.qwen_vl import QwenVLCaptionAdapter
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
from hcmai.data.enrichment.dataset_cli import add_dataset_arguments, dataset_overrides
from hcmai.data.stores.frame import FrameStore
from thundercompute.pipeline import LLMService


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

    frames = [
        frame.model_dump(mode="python")
        for frame in FrameStore.load(frames_path).iter_frames()
    ]
    order = [str(frame["frame_id"]) for frame in frames]

    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")

    output = Path(output_dir)
    captioner = captioner or QwenVLCaptionAdapter(config)
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

    # Checkpoint publication uses the same bundle boundary as final output.
    provisional = {
        **old,
        "artifact_version": config.enrichment_version,
        "source_artifact": "captions.parquet",
        ENRICHMENT_VERSION: config.enrichment_version,
        "effective_configuration": asdict(config),
        "dataset_root": str(root),
        "resolved_model_revision": resolved_revision,
        "frame_store_id": frame_store_id,
    }

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
        provisional,
        frame_store_id=frame_store_id,
        resolved_revision=resolved_revision,
    )

    if set(rows) != set(order):
        raise RuntimeError("caption processing did not retain one row per frame")

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
    write_caption_artifacts(output, order, rows, failures, manifest)
    return manifest

def main(argv: Sequence[str] | None = None) -> int:
    """Run caption enrichment through the configured inference gateway."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_ENRICHMENT_CONFIG)
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    add_dataset_arguments(parser)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    dataset = dataset_overrides(args)
    job = (
        CaptionJobConfig.from_yaml(args.config, dataset=dataset)
        if dataset is not None
        else CaptionJobConfig.from_yaml(args.config)
    )
    app_path = Path(args.app_config)
    settings = AppConfig.from_yaml(app_path) if app_path.is_file() else AppConfig()
    from hcmai.data.enrichment.caption.adapters.remote import RemoteCaptionAdapter

    service = LLMService.remote(
        os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
        settings.inference,
    )
    try:
        manifest = generate_captions(
            args.frames or job.frames_path,
            args.output or job.output_dir,
            job.caption,
            RemoteCaptionAdapter(service, job.caption),
            dataset_root=args.data_root or job.dataset_root,
            frame_store_id=job.frame_store_id,
        )
    finally:
        service.close()
    keys = "completed_count", "failed_count", "skipped_count", "retried_count"
    print({key: manifest[key] for key in keys})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
