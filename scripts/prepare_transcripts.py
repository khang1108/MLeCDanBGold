"""Build canonical transcript metadata from a video corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hcmai.common.config import ASRConfig, DiarizationConfig
from hcmai.data.enrichment.transcripts.pipeline import TranscriptService


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse transcript preparation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run transcription and print its compact final report."""

    args = parse_args(argv)
    service = TranscriptService.from_configs(ASRConfig(), DiarizationConfig())
    report = service.prepare(
        args.videos_root,
        args.output,
        resume=not args.no_resume,
        limit=args.limit,
    )
    print(f"Expected videos: {report.expected}")
    print(f"Transcribed: {report.transcribed}")
    print(f"No speech: {report.no_speech}")
    print(f"Failed: {len(report.failed)}")
    print(f"Segments: {report.segments}")
    print(f"Output: {report.output_path}")
    passed = report.expected > 0 and not report.failed
    print(f"Status: {'PASSED' if passed else 'FAILED'}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
