"""Run the resumable S3-first HCMAI corpus preparation pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hcmai.data.corpus_build import (
    S3CorpusPreparationConfig,
    S3CorpusPreparationService,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the intentionally small batch-job interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/preparation.s3.yaml",
    )
    parser.add_argument(
        "--enrichment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/enrichment.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "llm/config.yaml",
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=PROJECT_ROOT / "configs/baseline.yaml",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run preparation and print stable, automation-friendly completion data."""

    args = parse_args(argv)
    try:
        config = S3CorpusPreparationConfig.from_yaml(args.config)
        run = S3CorpusPreparationService(
            config,
            resume=not args.no_resume,
            limit=args.limit,
            enrichment_config=args.enrichment_config,
            model_config=args.model_config,
            retrieval_config=args.retrieval_config,
        ).run()
    except Exception as error:  # noqa: BLE001 - top-level batch failure boundary
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Run ID: {run.run_id}")
    print(f"S3 videos: {run.source_count}")
    print(f"Inventory: {run.inventory_path}")
    print(f"Artifacts: {run.artifacts_root}")
    print(f"Completed: {','.join(run.completed_stages) or '-'}")
    print(f"Resumed: {','.join(run.skipped_stages) or '-'}")
    if run.publication:
        print(f"Publication Bundle: {run.publication.bundle_id}")
        print(f"Publication Files: {run.publication.file_count}")
        print(f"Publication Size: {run.publication.total_bytes} bytes")
        print(f"Publication Latest: s3://{run.publication.bucket}/{run.publication.latest_key}")
    print("Status: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
