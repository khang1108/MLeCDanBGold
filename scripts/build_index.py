"""Build an exact FAISS visual index from generated embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmai.common.utils.io import read_parquet, read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retriever.index import VisualIndex
from script_args import parse_arguments

logger = get_logger(__name__)


def main() -> int:
    """Build and serialize one exact FAISS index."""
    args = parse_arguments("build_index")
    configure_logging(level=args.log_level)
    try:
        config = read_yaml(Path(args.config))
        embeddings = np.load(Path(args.embeddings))
        mapping = read_parquet(Path(args.mapping))
        model = config.get("models", {}).get("embedding", {})
        index = VisualIndex.build(
            embeddings,
            mapping,
            dataset_version=config.get("dataset", {}).get(
                "version", "unknown"
            ),
            model_name=model.get("name", "unknown"),
            index_type=config.get("index", {}).get("type", "flat_ip"),
        )
        index.save(Path(args.output))
        return 0
    except Exception as error:
        logger.exception("Index build failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
