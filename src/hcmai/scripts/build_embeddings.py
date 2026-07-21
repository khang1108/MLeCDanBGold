"""CLI script to generate frame embeddings using the embedding pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.embedding.embedding import EmbeddingPipeline
from hcmai.scripts.parse import parse_arguments

logger = get_logger(__name__)


def main() -> int:
    """Main entry point for the embedding generation script."""
    args = parse_arguments("build_embeddings")
    configure_logging(level=args.log_level, log_file=args.log_file)

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1

        config = read_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")

        frames_path = args.frames or config.get("dataset", {}).get("frames_path")
        if not frames_path:
            logger.error("frames_path not specified in config or arguments")
            return 1

        frames_path = Path(frames_path)
        if not frames_path.exists():
            logger.error(f"Frames file not found: {frames_path}")
            return 1

        encoder_config = EncoderConfig.from_dict(
            config.get("models", {}).get("embedding", {})
        )
        logger.info(f"Encoder config: {encoder_config}")

        pipeline = EmbeddingPipeline(
            frames_path=frames_path,
            output_dir=Path(args.output),
            encoder_config=encoder_config,
            dataset_version=config.get("dataset", {}).get("version", "unknown"),
        )
        metadata = pipeline.run()
        logger.info(f"Metadata: {metadata.to_dict()}")
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
