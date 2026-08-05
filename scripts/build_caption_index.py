"""Build caption, OCR, or ASR embeddings and an aligned FAISS index."""

from __future__ import annotations

import argparse

from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.common.schemas import RetrievalSource
from hcmai.retriever.pipeline import RetrievalService

logger = get_logger(__name__)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one frame-text source and an exact FAISS index"
    )
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--model-config", default="llm/config.yaml")
    parser.add_argument(
        "--source",
        default=RetrievalSource.CAPTION.value,
        choices=tuple(
            source.value
            for source in RetrievalSource
            if source != RetrievalSource.VISUAL
        ),
    )
    parser.add_argument("--enrichment")
    parser.add_argument("--captions", help=argparse.SUPPRESS)
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
        RetrievalService.build_text_artifacts(
            args.config,
            args.model_config,
            source=RetrievalSource(args.source),
            enrichment_path=args.enrichment or args.captions,
            frames_path=args.frames,
            output_dir=args.output,
        )
        return 0
    except Exception as error:
        logger.exception("%s index build failed: %s", args.source.upper(), error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
