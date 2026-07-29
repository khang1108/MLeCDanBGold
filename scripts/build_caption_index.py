"""Build caption embeddings and their aligned FAISS index."""

from __future__ import annotations

import argparse

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retriever.caption import build_caption_artifacts

logger = get_logger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caption embeddings and an exact FAISS index"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--model-config", default="llm/config.yaml")
    parser.add_argument("--captions")
    parser.add_argument("--frames")
    parser.add_argument("--output")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    configure_logging(args.log_level)
    try:
        build_caption_artifacts(
            args.config,
            args.model_config,
            captions_path=args.captions,
            frames_path=args.frames,
            output_dir=args.output,
        )
        return 0
    except Exception as error:
        logger.exception("Caption index build failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
