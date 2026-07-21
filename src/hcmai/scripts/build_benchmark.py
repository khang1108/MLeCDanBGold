"""CLI script to benchmark the dense-retrieval baseline on labelled queries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hcmai.common.schemas.evaluation import EvaluationQuery
from hcmai.common.utils.io import read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.retriever.benchmark import RetrievalBenchmark
from hcmai.retriever.dense import DenseRetriever
from hcmai.retriever.encoder import DenseEncoder, EncoderConfig
from hcmai.retriever.index import VisualIndex
from hcmai.scripts.parse import parse_arguments

logger = get_logger(__name__)


def main() -> int:
    """Load queries, run the dense retriever, and freeze baseline metrics."""
    args = parse_arguments("build_benchmark")
    configure_logging(level=args.log_level)

    try:
        config = read_yaml(Path(args.config))

        # Parse one EvaluationQuery per JSONL line, skipping blank lines.
        queries = [
            EvaluationQuery(**json.loads(line))
            for line in Path(args.queries).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info(f"Loaded {len(queries)} evaluation queries")

        encoder = DenseEncoder(EncoderConfig.from_dict(config.get("models", {}).get("embedding", {})))
        index = VisualIndex.load(Path(args.index))
        retriever = DenseRetriever(encoder, index)

        benchmark = RetrievalBenchmark(retriever, queries, output_dir=Path(args.output), top_k=args.top_k)
        metrics = benchmark.run()
        logger.info(f"Benchmark metrics: {metrics}")
        return 0

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
