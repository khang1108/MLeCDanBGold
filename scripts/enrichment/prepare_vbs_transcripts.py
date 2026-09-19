"""Prepare local VBS transcripts via direct audio upload to the model API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from hcmai.common.config import InferenceConfig, TranscriptJobConfig
from offline.enrichment.transcripts.adapters.upload import RemoteUploadASRAdapter
from offline.enrichment.transcripts.materialize import materialize_transcript_artifact
from offline.enrichment.transcripts.pipeline import TranscriptService
from llm.pipeline import LLMService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/vbs_prepare.yaml"))
    parser.add_argument("--inference-url", default=os.getenv("HCMAI_INFERENCE_BASE_URL", "http://127.0.0.1:8100"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    job = TranscriptJobConfig.from_yaml(args.config)
    inference = InferenceConfig(
        base_url=args.inference_url,
        timeout_seconds=120,
        connect_timeout_seconds=10,
        read_timeout_seconds=600,
        write_timeout_seconds=600,
        pool_timeout_seconds=10,
        max_attempts=2,
        max_concurrency=2,
    )
    remote = LLMService.remote(args.inference_url, inference)
    adapter = RemoteUploadASRAdapter(remote.adapter, job.asr)
    service = TranscriptService(adapter, None)
    videos_root = args.data / "videos"
    output = args.data / "artifacts" / "asr"
    try:
        report = service.prepare(
            videos_root,
            output,
            resume=not args.no_resume,
            limit=args.limit,
            schema_version=job.schema_version,
            pipeline_version=job.pipeline_version,
        )
    finally:
        remote.close()

    if not report.failed:
        materialize_transcript_artifact(
            args.data / "artifacts" / "frames.parquet",
            output,
            output / "frame_enrichment.parquet",
            window_ms=job.frame_evidence_window_ms,
            enrichment_version=job.enrichment_version,
            model_name=f"{job.asr.model_name}@{job.asr.revision}:{job.pipeline_version}",
            frame_store_id=job.frame_store_id or "vbs-local-v1",
        )
    print(f"Videos: {report.expected}")
    print(f"Transcribed: {report.transcribed}")
    print(f"No speech: {report.no_speech}")
    print(f"Failed: {len(report.failed)}")
    print(f"Segments: {report.segments}")
    return int(bool(report.failed))


if __name__ == "__main__":
    raise SystemExit(main())
