"""Shared argument parsing for root-level research scripts."""

from __future__ import annotations

import argparse

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
DESCRIPTIONS = {
    "build_embeddings": "Generate visual embeddings and their FAISS index",
    "build_index": "Build an exact FAISS index from visual embeddings",
    "build_benchmark": "Benchmark the dense-retrieval baseline",
}


def parse_arguments(command: str) -> argparse.Namespace:
    """Parse arguments for one supported root-level script."""
    parser = argparse.ArgumentParser(description=DESCRIPTIONS[command])
    parser.add_argument("--config", required=True)
    if command in {"build_embeddings", "build_index"}:
        parser.add_argument("--model-config", default="llm/config.yaml")
    if command == "build_embeddings":
        parser.add_argument("--frames")
        parser.add_argument("--dataset-root")
        parser.add_argument("--output", default="artifacts")
        parser.add_argument("--log-file")
    elif command == "build_index":
        parser.add_argument("--embeddings", required=True)
        parser.add_argument("--mapping", required=True)
        parser.add_argument("--output", required=True)
    elif command == "build_benchmark":
        parser.add_argument("--index", required=True)
        parser.add_argument("--queries", required=True)
        parser.add_argument("--output", default="runs/dense_model_comparison")
        parser.add_argument("--top-k", type=int, default=100)
    else:
        raise ValueError(f"Unknown command: {command!r}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
    )
    return parser.parse_args()
