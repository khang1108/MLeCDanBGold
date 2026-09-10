"""Evaluate saved HCMAI benchmark responses against temporal CSV labels.

The evaluator reports exact coordinate matches and a relaxed temporal match.
The latter reflects the competition's 1-fps keyframe sampling and avoids
penalizing a result that lands in the same event a few seconds away.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd


def load_labels(path: Path) -> list[tuple[str, ...]]:
    """Read headerless KIS or TRAKE ground-truth rows as strings."""

    with path.open(encoding="utf-8", newline="") as handle:
        return [tuple(row) for row in csv.reader(handle) if row]


def percentile(values: list[float], q: float) -> float | None:
    """Return a linear percentile, or ``None`` for an empty sample."""

    if not values:
        return None
    return float(pd.Series(values).quantile(q))


def evaluate(path: Path, query_root: Path, tolerance_frames: int = 150) -> dict[str, object]:
    """Evaluate all successful responses with a matching CSV label file."""

    responses = json.loads(path.read_text(encoding="utf-8"))
    mapping = pd.read_parquet("artifacts/frame_store/frames.parquet")
    frame_lookup = {
        str(row.frame_id): (str(row.video_id), int(row.frame_idx))
        for row in mapping.itertuples()
    }
    per_query: list[dict[str, object]] = []
    for item in responses:
        if item.get("status_code") != 200:
            continue
        query_path = Path(item["query_file"])
        label_path = query_root / query_path.parent.name / "ground_truth" / (
            query_path.stem + ".csv"
        )
        if not label_path.is_file():
            continue
        labels = load_labels(label_path)
        tolerance = 0 if tolerance_frames == 0 else tolerance_frames
        hits: list[bool] = []
        event_hits: list[list[bool]] = []
        body = item["response"]
        candidates = body.get("paths", []) if item["type"] == "trake" else body.get("results", [])
        for candidate in candidates:
            if item["type"] == "trake":
                video = str(candidate["video_id"])
                values = [int(value) for value in candidate["frame_idxs"]]
                matched_rows = [
                    row
                    for row in labels
                    if row and row[0] == video and len(row) == len(values) + 1
                ]
                per_event = [
                    any(abs(values[index] - int(row[index + 1])) <= tolerance for row in matched_rows)
                    for index in range(len(values))
                ]
                event_hits.append(per_event)
                hits.append(any(all(abs(values[i] - int(row[i + 1])) <= tolerance for i in range(len(values))) for row in matched_rows))
            else:
                pairs = [frame_lookup.get(str(fid)) for fid in candidate.get("frame_ids", [])]
                pairs = [pair for pair in pairs if pair is not None]
                hits.append(any(
                    any(pair[0] == row[0] and abs(pair[1] - int(row[1])) <= tolerance for pair in pairs)
                    for row in labels
                ))
        rank = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        per_query.append({
            "query_file": str(query_path),
            "type": item["type"],
            "label_count": len(labels),
            "candidate_count": len(hits),
            "first_hit_rank": rank,
            "hits": hits,
            "event_hits": event_hits,
            "latency_ms": body.get("latency", {}).get("total_ms"),
            "wall_s": item.get("elapsed_wall_s"),
        })

    summary: dict[str, object] = {"queries": len(per_query), "per_query": per_query}
    for kind in ("kis", "trake"):
        rows = [row for row in per_query if row["type"] == kind]
        if not rows:
            continue
        ranks = [row["first_hit_rank"] for row in rows if row["first_hit_rank"] is not None]
        metrics: dict[str, object] = {
            "queries": len(rows),
            "hit_rate": len(ranks) / len(rows),
            "mrr": sum(1 / rank for rank in ranks) / len(rows),
            "median_first_hit_rank": percentile([float(rank) for rank in ranks], 0.5),
            "mean_latency_ms": sum(float(row["latency_ms"] or 0) for row in rows) / len(rows),
            "p95_latency_ms": percentile([float(row["latency_ms"] or 0) for row in rows], 0.95),
        }
        for k in (1, 5, 10, 20, 50, 100):
            metrics[f"recall@{k}"] = sum(
                any(row["hits"][:k]) for row in rows
            ) / len(rows)
        if kind == "trake":
            all_events = [flag for row in rows for flags in row["event_hits"] for flag in flags]
            metrics["event_recall"] = sum(all_events) / len(all_events) if all_events else 0.0
        summary[kind] = metrics
    return summary


def main() -> None:
    """Evaluate exact and ±5-second (150-frame) temporal matching."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--query-root", default="artifacts/query")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    exact = evaluate(Path(args.responses), Path(args.query_root), tolerance_frames=0)
    relaxed = evaluate(Path(args.responses), Path(args.query_root), tolerance_frames=150)
    result = {"exact": exact, "relaxed_5s": relaxed}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, summary in result.items():
        print(label)
        for kind in ("kis", "trake"):
            if kind in summary:
                print(kind, json.dumps(summary[kind], ensure_ascii=False))


if __name__ == "__main__":
    main()
