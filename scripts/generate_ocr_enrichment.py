"""Build OCR enrichment using local Florence-2 or a remote inference gateway."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.data.enrichment.pipeline import EnrichmentJobConfig, EnrichmentService

logger = get_logger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/enrichment.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse config plus optional path and batch-size overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--frame-store-id")
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the OCR job, select an adapter, and delegate generation."""

    args = parse_args(argv)
    configure_logging(args.log_level)

    job = EnrichmentJobConfig.from_yaml(args.config)
    config = (
        replace(job.ocr, batch_size=args.batch_size)
        if args.batch_size is not None
        else job.ocr
    )
    engine = None

    app_path = Path(args.app_config)
    if app_path.exists():
        settings = AppConfig.from_yaml(app_path)
        base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
        if settings.inference.enabled or base_url:
            from hcmai.data.enrichment.ocr.adapters.remote import RemoteOCRAdapter
            from hcmai.llm.pipeline import LLMService

            client = LLMService.remote(base_url, settings.inference.timeout_seconds)
            engine = RemoteOCRAdapter(client.adapter, config)
            logger.info("Using remote OCR backend at %s", base_url)
        else:
            logger.info("Using local Florence-2 OCR backend")
    else:
        logger.info("No app config found; using local Florence-2 OCR backend")

    output = args.output or job.ocr_output_dir
    report = EnrichmentService.generate_ocr(
        args.frames or job.frames_path,
        output,
        config,
        engine,
        dataset_root=args.dataset_root or job.data_root,
        frame_store_id=args.frame_store_id or job.frame_store_id,
    )
    completed = report.get("completed_frames", 0)
    failed = report.get("failed_frames", 0)
    skipped = report.get("skipped_frames", 0)
    status = "DEGRADED" if failed else "COMPLETE"
    log_result = logger.warning if failed else logger.info
    log_result(
        "OCR %s: completed=%d failed=%d skipped=%d output=%s",
        status,
        completed,
        failed,
        skipped,
        output,
    )
    # Per-frame failures are represented in the successfully published bundle.
    # Artifact-level failures still propagate and produce a nonzero process exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
