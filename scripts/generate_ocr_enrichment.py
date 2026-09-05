"""Build OCR enrichment through the configured inference gateway."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.utils.logging import configure_logging, get_logger
from llm.pipeline import LLMService
from offline.enrichment.dataset_cli import (
    add_dataset_arguments,
    apply_overrides,
    dataset_overrides,
)
from offline.enrichment.ocr.adapters.remote import RemoteOCRAdapter
from offline.enrichment.pipeline import EnrichmentJobConfig, EnrichmentService

logger = get_logger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/prepare.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse config plus optional path and batch-size overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    add_dataset_arguments(parser)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the OCR job and delegate generation to the inference gateway."""

    args = parse_args(argv)
    configure_logging(args.log_level)

    job = EnrichmentJobConfig.from_yaml(args.config, dataset=dataset_overrides(args))
    config = apply_overrides(job.ocr, batch_size=args.batch_size)
    app_path = Path(args.app_config)
    settings = AppConfig.from_yaml(app_path) if app_path.is_file() else AppConfig()
    base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
    service = LLMService.remote(base_url, settings.inference)
    engine = RemoteOCRAdapter(service, config)
    logger.info("Using OCR inference gateway at %s", base_url)

    output = args.output or job.ocr_output_dir
    try:
        report = EnrichmentService.generate_ocr(
            args.frames or job.frames_path,
            output,
            config,
            engine,
            dataset_root=args.data_root or job.data_root,
            frame_store_id=args.frame_store_id or job.frame_store_id,
        )
    finally:
        service.close()
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
