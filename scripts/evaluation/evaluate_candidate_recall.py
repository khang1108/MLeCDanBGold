#!/usr/bin/env python3
"""Run Candidate Recall@K evaluation across candidate budgets K.

Usage:
    python scripts/evaluation/evaluate_candidate_recall.py --diagnostic-file data/eval/transition_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcmai.temporal.candidate_recall import (
    EventAnnotation,
    evaluate_candidate_recall,
    format_candidate_recall_table,
)


def load_diagnostic_annotations(path: Path | str) -> list[EventAnnotation]:
    """Load event annotations from diagnostic JSON dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    annotations: list[EventAnnotation] = []
    items = data if isinstance(data, list) else data.get("queries", [])
    for item in items:
        video_id = item["video_id"]
        for idx, ev in enumerate(item.get("events", [])):
            target_idxs = tuple(ev.get("target_frame_idxs", ()))
            start_ms = ev.get("start_ms")
            end_ms = ev.get("end_ms")
            annotations.append(
                EventAnnotation(
                    video_id=video_id,
                    event_index=idx,
                    target_frame_idxs=target_idxs,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
            )
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Candidate Recall@K")
    parser.add_argument(
        "--diagnostic-file",
        type=str,
        default="data/eval/transition_diagnostic.json",
        help="Path to diagnostic dataset JSON",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Candidate budgets K to evaluate",
    )
    args = parser.parse_args()

    diag_path = Path(args.diagnostic_file)
    if not diag_path.is_file():
        print(f"Diagnostic file {diag_path} not found. Exiting.")
        return

    annotations = load_diagnostic_annotations(diag_path)
    print(f"Loaded {len(annotations)} event annotations from {diag_path}.")


if __name__ == "__main__":
    main()
