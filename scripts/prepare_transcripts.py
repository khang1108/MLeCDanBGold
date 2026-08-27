"""Build canonical transcript metadata from a video corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hcmai.common.config import TranscriptJobConfig
from hcmai.data.enrichment.dataset_cli import add_dataset_arguments, dataset_overrides
from hcmai.data.enrichment.transcripts.materialize import (
    materialize_transcript_artifact,
)
from hcmai.data.enrichment.transcripts.pipeline import TranscriptService

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/prepare.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse transcript preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_dataset_arguments(parser)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-enrichment-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-diarization", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run transcription and print its compact final report."""

    args = parse_args(argv)
    dataset = dataset_overrides(args)
    if dataset is None:
        raise ValueError(
            "transcript preparation requires the complete dataset CLI contract"
        )
    config = TranscriptJobConfig.from_yaml(args.config, dataset=dataset)
    if args.no_diarization:
        config = config.model_copy(update={
            "diarization": config.diarization.model_copy(update={"enabled": False})
        })
    output = args.output or config.output_dir
    service = TranscriptService.from_job_config(config)
    report = service.prepare(
        args.videos_root,
        output,
        resume=not args.no_resume,
        limit=args.limit,
        schema_version=config.schema_version,
        pipeline_version=config.pipeline_version,
    )
    frame_output = args.frame_enrichment_output or config.frame_enrichment_path
    if not report.failed:
        materialize_transcript_artifact(
            args.frames or config.frames_path,
            output,
            frame_output,
            window_ms=config.frame_evidence_window_ms,
            enrichment_version=config.enrichment_version,
            model_name=(
                f"{config.asr.model_name}@{config.asr.revision}:"
                f"{config.pipeline_version}"
            ),
            frame_store_id=config.frame_store_id,
        )
    print(f"Expected videos: {report.expected}")
    print(f"Transcribed: {report.transcribed}")
    print(f"No speech: {report.no_speech}")
    print(f"Failed: {len(report.failed)}")
    print(f"Segments: {report.segments}")
    print(f"Output: {report.output_path}")
    if not report.failed:
        print(f"Frame enrichment: {frame_output}")
    passed = report.expected > 0 and not report.failed
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
