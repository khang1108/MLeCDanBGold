"""Run a reproducible dense-retrieval baseline and record its metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from hcmai.common.schemas.evaluation import EvaluationQuery
from hcmai.common.utils.io import write_json, write_yaml
from hcmai.common.utils.logging import get_logger
from hcmai.common.utils.timing import Timer
from hcmai.retriever.pipeline import RetrievalService
from hcmai.common.config import RECALL_CUTOFFS, BenchmarkConfig

logger = get_logger(__name__)


class RetrievalBenchmark:
    """Evaluate a dense retriever on labelled queries and write reports.

    Emits the four frozen baseline artifacts under the output directory:
    ``config.yaml``, ``metrics.json``, ``per_query.csv``, and ``failures.csv``.
    """

    def __init__(
        self,
        retriever: RetrievalService,
        queries: list[EvaluationQuery],
        output_dir: Path | str,
        *,
        run_name: str = "dense_baseline",
        top_k: int = 100,
    ) -> None:
        self.retriever = retriever
        self.queries = queries
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.top_k = top_k

    def run(self) -> dict[str, Any]:
        """Run every query, compute metrics, and write baseline artifacts.

        Returns:
            The aggregate metrics dictionary that is also written to
            ``metrics.json``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.retriever.index_metadata

        # Reset GPU peak memory tracking so the reported figure reflects only
        # this benchmark run; ignored when CUDA is unavailable.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None

        per_query_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        latencies_ms: list[float] = []

        for query in self.queries:
            timer = Timer()
            result = self.retriever.search(query.query, top_k=self.top_k)
            candidates = result.candidates
            total_ms = timer.stop()
            latencies_ms.append(total_ms)

            retrieved_ids = [candidate.frame_id for candidate in candidates]
            gold = set(query.gold_frame_ids)

            # Recall@k = fraction of this query's gold frames found within the
            # top-k retrieved results; computed once per cut-off inline.
            recalls: dict[int, float] = {}
            for cutoff in RECALL_CUTOFFS:
                hits = gold.intersection(retrieved_ids[:cutoff])
                recalls[cutoff] = len(hits) / len(gold)

            # A query with no gold frame anywhere in the ranked list is a
            # retrieval failure worth documenting for error analysis.
            first_gold_rank = next(
                (
                    rank
                    for rank, frame_id in enumerate(retrieved_ids, start=1)
                    if frame_id in gold
                ),
                None,
            )
            if first_gold_rank is None:
                failure_rows.append(
                    {
                        "query_id": query.query_id,
                        "query": query.query,
                        "task_type": query.task_type.value,
                        "difficulty": query.difficulty.value,
                        "num_gold": len(gold),
                        "num_retrieved": len(retrieved_ids),
                    }
                )

            per_query_rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "task_type": query.task_type.value,
                    "difficulty": query.difficulty.value,
                    "first_gold_rank": first_gold_rank,
                    "query_encoding_ms": result.trace.duration_for(
                        "query_encoding"
                    ),
                    "index_search_ms": result.trace.duration_for("index_search"),
                    "total_ms": total_ms,
                    **{f"recall@{k}": recalls[k] for k in RECALL_CUTOFFS},
                }
            )

        latencies = np.asarray(latencies_ms, dtype=np.float64)
        gpu_peak_bytes = 0
        if torch is not None and torch.cuda.is_available():
            gpu_peak_bytes = int(torch.cuda.max_memory_allocated())

        metrics = {
            "run_name": self.run_name,
            "num_queries": len(self.queries),
            "recall": {
                f"@{k}": float(
                    np.mean([row[f"recall@{k}"] for row in per_query_rows])
                    if per_query_rows
                    else 0.0
                )
                for k in RECALL_CUTOFFS
            },
            "throughput_qps": float(
                len(self.queries) / (latencies.sum() / 1000.0)
                if latencies.sum() > 0
                else 0.0
            ),
            "latency_ms": {
                "p50": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
                "p95": float(np.percentile(latencies, 95)) if latencies.size else 0.0,
                "mean": float(latencies.mean()) if latencies.size else 0.0,
            },
            "index_build_time_sec": metadata.build_time_sec,
            "index_size_bytes": metadata.index_size_bytes,
            "gpu_peak_memory_bytes": gpu_peak_bytes,
            "num_failures": len(failure_rows),
        }

        # Write the four frozen baseline artifacts.
        config = BenchmarkConfig(
            run_name=self.run_name,
            dataset_version=metadata.dataset_version,
            model_name=metadata.model_name,
            index_type=metadata.index_type,
            num_queries=len(self.queries),
            top_k=self.top_k,
        )
        write_yaml(config.__dict__, self.output_dir / "config.yaml")
        write_json(metrics, self.output_dir / "metrics.json")
        pd.DataFrame(per_query_rows).to_csv(
            self.output_dir / "per_query.csv", index=False
        )
        pd.DataFrame(failure_rows).to_csv(
            self.output_dir / "failures.csv", index=False
        )

        logger.info(
            f"Benchmark '{self.run_name}': "
            f"R@1={metrics['recall']['@1']:.3f}, "
            f"R@20={metrics['recall']['@20']:.3f}, "
            f"R@100={metrics['recall']['@100']:.3f}, "
            f"P95={metrics['latency_ms']['p95']:.1f}ms, "
            f"{metrics['throughput_qps']:.1f} qps, "
            f"failures={metrics['num_failures']}"
        )
        return metrics
