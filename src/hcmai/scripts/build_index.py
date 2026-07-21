"""CLI script to build a FAISS visual index from generated embeddings."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

from hcmai.common.utils.io import read_yaml, read_parquet
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retriever.index import VisualIndex
from hcmai.scripts.parse import parse_arguments

logger = get_logger(__name__)


def main() -> int:
    """Build and serialize an exact FAISS index for a version's embeddings."""
    args = parse_arguments("build_index")
    configure_logging(level=args.log_level)

    try:
        config = read_yaml(Path(args.config))

        embeddings = np.load(Path(args.embeddings))
        mapping = read_parquet(Path(args.mapping))
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(mapping)} mapping rows")

        index = VisualIndex.build(
            embeddings,
            mapping,
            dataset_version=config.get("dataset", {}).get("version", "unknown"),
            model_name=config.get("models", {}).get("embedding", {}).get("name", "unknown"),
            index_type=config.get("index", {}).get("type", "flat_ip"),
        )
        index.save(Path(args.output))

        logger.info(f"Index metadata: {index.metadata.to_dict()}")
        return 0

    except Exception as e:
        logger.exception(f"Index build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
