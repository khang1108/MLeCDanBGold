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
from hcmai.enrichment.caption.artifacts import write_caption_artifacts
from hcmai.enrichment.caption.backend import CaptionBackend, FrameCaptioner
from hcmai.enrichment.caption.config import (
    DEFAULT_ENRICHMENT_CONFIG,
    CaptionConfig,
    CaptionJobConfig,
    ENRICHMENT_VERSION,
)
from hcmai.enrichment.caption.report import build_manifest
from hcmai.enrichment.caption.resume import guard_resume, resume_rows
from hcmai.enrichment.caption.runner import run_batches
from hcmai.llm.config import LLMServiceConfig

def generate_captions(
    frames_path: str | Path,
    output_dir: str | Path,
    config: CaptionConfig,
    captioner: CaptionBackend | None = None,
    *,
    dataset_root: str | Path = "."
) -> dict[str, Any]:
    """Generate or resume one deterministic caption enrichment artifact."""
    started, began, frames_path = datetime.now(timezone.utc), perf_counter(), Path(frames_path)
    root = Path(dataset_root).expanduser().resolve()
    frames = cast(
        list[dict[str, Any]],
        pd.read_parquet(frames_path).to_dict(orient="records"),
    )
    order = [frame["frame_id"] for frame in frames]
    if len(order) != len(set(order)):
        raise ValueError("input frames contain duplicate frame_id values")
    output, captioner = Path(output_dir), captioner or FrameCaptioner(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, parquet_path = output / "manifest.json", output / "frame_enrichment.parquet"
    old = read_json(manifest_path) if manifest_path.exists() else {}
    guard_resume(parquet_path, old, config, root)
    rows, todo, skipped, retried = resume_rows(frames, parquet_path, config)
    resolved_revision = captioner.resolve_revision()
    guard_resume(parquet_path, old, config, root, resolved_revision)
    provisional = {
        **old,
        ENRICHMENT_VERSION: config.enrichment_version,
        "effective_configuration": asdict(config),
        "dataset_root": str(root),
        "resolved_model_revision": resolved_revision,
    }
    atomic_write(manifest_path, lambda path: write_json(provisional, path))
    failures: dict[str, dict[str, str]] = {}
    latencies = run_batches(
        todo, order, rows, failures, captioner, config, output, root
    )
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
        from hcmai.llm.client import InferenceClient, RemoteFrameCaptioner

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
        client = InferenceClient(
            os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url),
            settings.inference.timeout_seconds,
        )
        captioner = RemoteFrameCaptioner(client, job.caption)
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
