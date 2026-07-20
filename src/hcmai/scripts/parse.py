"""Argument parsing for the ``build_*`` CLI scripts.

A single :func:`parse_arguments` builds the right parser for each command so
every ``build_*.py`` script holds only its run logic.
"""

from __future__ import annotations

import argparse

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
DESCRIPTIONS = {
    "build_embeddings": "Generate frame embeddings for the retrieval corpus",
    "build_index": "Build an exact FAISS IndexFlatIP from visual embeddings",
    "build_benchmark": "Benchmark the dense-retrieval baseline",
}


def parse_arguments(command: str) -> argparse.Namespace:
    """Parse command-line arguments for the given ``build_*`` command."""
    parser = argparse.ArgumentParser(description=DESCRIPTIONS[command])
    parser.add_argument("--config", type=str, required=True, help="Path to baseline.yaml")

    if command == "build_embeddings":
        parser.add_argument("--frames", type=str, help="Override frames path from config")
        parser.add_argument(
            "--output", type=str, default="artifacts", help="Output directory (default: artifacts)"
        )
        parser.add_argument("--log-file", type=str, help="Optional log file path")

    elif command == "build_index":
        parser.add_argument("--embeddings", type=str, required=True, help="Path to visual_embeddings.npy")
        parser.add_argument("--mapping", type=str, required=True, help="Path to frame_mapping.parquet")
        parser.add_argument("--output", type=str, required=True, help="Output directory for the index")

    elif command == "build_benchmark":
        parser.add_argument("--index", type=str, required=True, help="Directory with the FAISS index")
        parser.add_argument("--queries", type=str, required=True, help="JSONL of EvaluationQuery records")
        parser.add_argument(
            "--output", type=str, default="runs/dense_model_comparison", help="Output directory"
        )
        parser.add_argument("--top-k", type=int, default=100, help="Retrieval depth")

    else:
        raise ValueError(f"Unknown command: {command!r}")

    parser.add_argument("--log-level", type=str, default="INFO", choices=LOG_LEVELS, help="Logging level")
    return parser.parse_args()
