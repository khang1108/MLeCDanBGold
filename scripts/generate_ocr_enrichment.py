"""Build OCR enrichment using local Florence-2 or a remote inference gateway."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.enrichment.ocr.config import OCRConfig
from hcmai.enrichment.ocr.generator import generate_ocr

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", default="data/metadata/frames.parquet", type=Path
    )
    parser.add_argument(
        "--output", default="artifacts/enrichment/ocr", type=Path
    )
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--app-config", default="configs/baseline.yaml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    config = OCRConfig(batch_size=args.batch_size)
    engine = None

    app_path = Path(args.app_config)
    if app_path.exists():
        settings = AppConfig.from_yaml(app_path)
        base_url = os.getenv("HCMAI_INFERENCE_BASE_URL", settings.inference.base_url)
        if settings.inference.enabled or base_url:
            from hcmai.enrichment.ocr.adapters.remote import RemoteOCRAdapter
            from hcmai.llm.pipeline import LLMService

            client = LLMService.remote(base_url, settings.inference.timeout_seconds)
            engine = RemoteOCRAdapter(client, config)
            logger.info("Using remote OCR backend at %s", base_url)
        else:
            logger.info("Using local Florence-2 OCR backend")
    else:
        logger.info("No app config found; using local Florence-2 OCR backend")

    report = generate_ocr(
        args.frames,
        args.output,
        config,
        engine=engine,
        dataset_root=args.dataset_root,
    )
    completed = report.get("completed_frames", 0)
    failed = report.get("failed_frames", 0)
    skipped = report.get("skipped_frames", 0)
    logger.info(
        "OCR done: completed=%d failed=%d skipped=%d output=%s",
        completed, failed, skipped, args.output,
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
