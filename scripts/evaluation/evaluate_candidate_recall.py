#!/usr/bin/env python3
"""Run Candidate Recall@K evaluation across candidate budgets K.

Usage:
    python scripts/evaluation/evaluate_candidate_recall.py --diagnostic-file data/eval/transition_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from hcmai.temporal.candidate_recall import (
    EventAnnotation,
    evaluate_candidate_recall,
    format_candidate_recall_table,
)
from hcmai.temporal.diagnostic import DiagnosticDataset
from scripts.evaluation.eval_corpus_builder import build_synthetic_eval_corpus


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
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/candidate_recall.json",
        help="Output path for Candidate Recall JSON record",
    )
    args = parser.parse_args()

    diag_path = Path(args.diagnostic_file)
    if not diag_path.is_file():
        print(f"Diagnostic file {diag_path} not found. Exiting.")
        return

    annotations = load_diagnostic_annotations(diag_path)
    dataset = DiagnosticDataset.load_json(diag_path)
    print(f"Loaded {len(annotations)} event annotations across {len(dataset)} queries from {diag_path}.")

    # Build evaluation video scores
    video_scores, _ = build_synthetic_eval_corpus(dataset)

    results = evaluate_candidate_recall(
        annotations=annotations,
        video_scores=video_scores,
        k_values=args.k_values,
    )

    print("\n" + format_candidate_recall_table(results))

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dict = {
        "recalls": {str(r.k): r.recall for r in results},
        "details": [
            {
                "k": r.k,
                "recall": r.recall,
                "total_events": r.total_events,
                "hit_events": r.hit_events,
                "avg_edges_per_video": r.avg_edges_per_video,
                "avg_latency_ms": r.avg_latency_ms,
            }
            for r in results
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\nCandidate Recall results saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
