#!/usr/bin/env python3
"""Run P0 temporal evidence ablations (A0-A5) and record ranking telemetry.

Usage:
    PYTHONPATH=src:. aic/bin/python scripts/evaluate_temporal_p0.py \\
        --queries-file tests/fixtures/l26_v254_query.yaml \\
        --runs A0 A1 A2 A3 A4 A5 \\
        --output-file artifacts/p0_ablation_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import yaml

from hcmai.orchestration.setup import load_search_service
from hcmai.retrieval.evidence.ablation import (
    ABLATION_RUNS,
    AblationRunConfig,
    resolve_ablation_run,
)
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P0 temporal evidence ablation matrix A0-A5"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("tests/fixtures/l26_v254_query.yaml"),
        help="Path to YAML/JSON query file (default: tests/fixtures/l26_v254_query.yaml)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["A0", "A1", "A2", "A3", "A4", "A5"],
        help="List of ablation runs to execute (e.g. A0 A1 ... or full names)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("artifacts/p0_ablation_results.jsonl"),
        help="Path to output JSONL file (default: artifacts/p0_ablation_results.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=300,
        help="Number of paths to rank in DP alignment (default: 300)",
    )
    parser.add_argument(
        "--use-dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dense scoring",
    )
    parser.add_argument(
        "--use-bm25",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable BM25 scoring",
    )
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load query items from YAML or JSON."""

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
        items = data["queries"]
    elif isinstance(data, dict):
        items = [data]
    else:
        raise ValueError(f"Unrecognized query file format in {path}")

    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        raw_query = item.get("query", "")
        if isinstance(raw_query, str):
            orig_events = [line.strip() for line in raw_query.strip().splitlines() if line.strip()]
        elif isinstance(raw_query, list):
            orig_events = [str(x).strip() for x in raw_query if str(x).strip()]
        else:
            orig_events = []

        ret_events = item.get("retrieval_events", orig_events)
        cap_events = item.get("caption_events", orig_events)
        target_video = item.get("target_video_id")
        if not target_video and "l26_v254" in str(path).lower():
            target_video = "L26_V254"

        query_id = item.get("query_id", f"q{idx + 1}")
        parsed.append(
            {
                "query_id": query_id,
                "original_events": orig_events,
                "retrieval_events": ret_events,
                "caption_events": cap_events,
                "target_video_id": target_video,
            }
        )
    return parsed


def main() -> int:
    args = parse_args()

    if not args.queries_file.is_file():
        print(f"Error: query file not found: {args.queries_file}", file=sys.stderr)
        return 1

    queries = load_queries(args.queries_file)
    if not queries:
        print(f"Error: no queries found in {args.queries_file}", file=sys.stderr)
        return 1

    runs: list[AblationRunConfig] = [resolve_ablation_run(r) for r in args.runs]

    print(f"Loaded {len(queries)} query items. Ablation runs: {[r.name for r in runs]}")

    messages: list[str] = []
    service = load_search_service(messages)
    for msg in messages:
        print(f"[{msg}]", file=sys.stderr)

    temporal_evidence = getattr(service, "temporal_evidence", None)
    temporal_search = (
        getattr(service, "temporal_search", None)
        or getattr(service, "temporal", None)
        or (getattr(service.trake, "temporal", None) if hasattr(service, "trake") else None)
        or (getattr(service.kis, "temporal", None) if hasattr(service, "kis") else None)
    )

    if not isinstance(temporal_evidence, TemporalEvidenceScorer) or temporal_search is None:
        print(
            f"Error: temporal search service not properly configured "
            f"(evidence={type(temporal_evidence)}, search={type(temporal_search)})",
            file=sys.stderr,
        )
        return 1

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 105)
    print(
        f"{'Run':<22} {'Query':<14} {'Target':<10} {'Rank':<6} {'TargetScore':<12} "
        f"{'TopVideo':<10} {'TopScore':<10} {'ScoreGap':<10} {'Retr(ms)':<10} {'Align(ms)'}"
    )
    print("-" * 105)

    for run_cfg in runs:
        # Apply run configuration to temporal evidence scorer
        temporal_evidence.config = run_cfg.to_hybrid_config()

        # Update ASR index interval projection setting
        asr_index = getattr(temporal_evidence.dense, "asr_index", None)
        if asr_index is not None and hasattr(asr_index, "interval_projection"):
            asr_index.interval_projection = run_cfg.interval_projection

        # Apply alignment config (ensuring identity across all stages)
        temporal_search.config = run_cfg.alignment

        for q in queries:
            search_result = temporal_search.search(
                original_events=q["original_events"],
                retrieval_events=q["retrieval_events"],
                caption_events=q["caption_events"],
                top_k=args.top_k,
                use_dense=args.use_dense,
                use_bm25=args.use_bm25,
            )

            paths = search_result.paths
            top_video_id = paths[0].video_id if paths else None
            top_score = float(paths[0].score) if paths else 0.0
            top_10_videos = [p.video_id for p in paths[:10]]

            target_video_id = q.get("target_video_id")
            target_rank: int | None = None
            target_score: float | None = None
            target_frames: list[int] | None = None

            if target_video_id:
                for rank, path in enumerate(paths, start=1):
                    if path.video_id == target_video_id:
                        target_rank = rank
                        target_score = float(path.score)
                        raw_frames = getattr(path, "frame_idxs", getattr(path, "frame_idx", ()))
                        target_frames = [int(x) for x in raw_frames]
                        break

            score_gap = round(top_score - target_score, 4) if target_score is not None else None

            row: dict[str, Any] = {
                "run": run_cfg.name,
                "query_id": q["query_id"],
                "top_video_id": top_video_id,
                "top_score": round(top_score, 4),
                "top_10_videos": top_10_videos,
                "retrieval_ms": round(search_result.retrieval_ms, 2),
                "alignment_ms": round(search_result.alignment_ms, 2),
            }
            if target_video_id:
                row["target_video_id"] = target_video_id
                if target_rank is not None:
                    row["target_rank"] = target_rank
                    row["target_score"] = round(target_score, 4)
                    row["score_gap"] = score_gap
                    row["target_frame_indices"] = target_frames

            results.append(row)

            rank_display = str(target_rank) if target_rank is not None else f">{args.top_k}"
            target_score_disp = f"{target_score:.3f}" if target_score is not None else "N/A"
            gap_disp = f"{score_gap:.3f}" if score_gap is not None else "N/A"
            target_display = target_video_id or "N/A"
            top_display = top_video_id or "N/A"
            print(
                f"{run_cfg.name:<22} {q['query_id']:<14} {target_display:<10} "
                f"{rank_display:<6} {target_score_disp:<12} {top_display:<10} {top_score:<10.3f} "
                f"{gap_disp:<10} {search_result.retrieval_ms:<10.1f} {search_result.alignment_ms:<10.1f}"
            )
            if target_frames is not None:
                print(f"   Target aligned frame indices: {target_frames}")
            print(f"   Top-10 video IDs: {', '.join(top_10_videos)}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} ablation records to {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
