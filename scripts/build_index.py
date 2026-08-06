"""Build an exact FAISS visual index from generated embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmai.common.utils.io import read_parquet, read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.llm.pipeline import LLMServiceConfig
from hcmai.retriever.pipeline import RetrievalService
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
        models = LLMServiceConfig.from_yaml(args.model_config)
        index = RetrievalService.build_index(
            embeddings,
            mapping,
            dataset_version=config.get("dataset", {}).get(
                "version", "unknown"
            ),
            model_name=models.visual_embedding.model_name,
            index_type=config.get("index", {}).get("type", "flat_ip"),
        )
        index.save(Path(args.output))
        return 0
    except Exception as error:
        logger.exception("Index build failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
