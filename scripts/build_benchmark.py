"""Benchmark the dense-retrieval baseline on labelled queries."""

from __future__ import annotations

import json
from pathlib import Path

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas.evaluation import EvaluationQuery
from hcmai.common.utils.io import read_yaml
from hcmai.common.utils.logging import configure_logging, get_logger
from hcmai.embedding.pipeline import EmbeddingService
from hcmai.retriever.evaluation import RetrievalBenchmark
from hcmai.retriever.pipeline import RetrievalService
from script_args import parse_arguments

logger = get_logger(__name__)


def main() -> int:
    """Run the configured dense-retrieval benchmark."""
    args = parse_arguments("build_benchmark")
    configure_logging(level=args.log_level)
    try:
        config = read_yaml(Path(args.config))
        lines = Path(args.queries).read_text(encoding="utf-8").splitlines()
        queries = [
            EvaluationQuery(**json.loads(line))
            for line in lines
            if line.strip()
        ]
        model = config.get("models", {}).get("embedding", {})
        encoder = EmbeddingService.create_text_adapter(
            EncoderConfig.from_dict(model)
        )
        retriever = RetrievalService.from_index(
            RetrievalService.load_index(Path(args.index)),
            encoder,
        )
        benchmark = RetrievalBenchmark(
            retriever,
            queries,
            output_dir=Path(args.output),
            top_k=args.top_k,
        )
        logger.info("Benchmark metrics: %s", benchmark.run())
        return 0
    except Exception as error:
        logger.exception("Benchmark failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
