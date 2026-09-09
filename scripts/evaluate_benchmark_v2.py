"""Evaluate HCMAI API responses with identity- and time-aware metrics.

Ground-truth CSVs contain official ``frame_idx`` coordinates.  The runtime
corpus is a custom 1-fps timeline, so relaxed matching converts both sides to
milliseconds using the canonical per-video FPS.  Missing CSVs are reported as
unlabeled and are never counted as failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


K_VALUES = (1, 5, 10, 20, 50, 100)


def labels_from(path: Path) -> list[tuple[str, ...]]:
    """Read a headerless ground-truth file, tolerating a UTF-8 BOM."""

    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [tuple(row) for row in csv.reader(handle) if row]


def wilson(successes: int, total: int) -> list[float] | None:
    """Return a two-sided 95% Wilson interval for a proportion."""

    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def rank_of(values: list[bool]) -> int | None:
    """Return one-based first-hit rank, or ``None`` when absent."""

    return next((i + 1 for i, value in enumerate(values) if value), None)


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a percentile for a non-empty list."""

    return float(pd.Series(values).quantile(fraction)) if values else None


def load_frame_maps(path: Path) -> tuple[dict[str, tuple[str, int, int]], dict[str, float]]:
    """Load ``frame_id -> (video, frame_idx, timestamp_ms)`` and FPS maps."""

    frame_map: dict[str, tuple[str, int, int]] = {}
    fps_map: dict[str, float] = {}
    table = pd.read_parquet(path)
    for row in table.itertuples():
        video = str(row.video_id)
        frame_map[str(row.frame_id)] = (video, int(row.frame_idx), int(row.timestamp_ms))
        if row.fps is not None and not pd.isna(row.fps):
            fps_map.setdefault(video, float(row.fps))
    return frame_map, fps_map


def label_ms(video: str, frame_idx: int, fps_map: dict[str, float]) -> float:
    """Convert an official frame coordinate to milliseconds."""

    return frame_idx * 1000.0 / fps_map.get(video, 30.0)


def candidate_frames(candidate: dict[str, Any], frame_map: dict[str, tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Resolve representative and aligned path frame IDs in candidate order."""

    values: list[tuple[str, int, int]] = []
    for raw_id in candidate.get("frame_ids", []):
        value = frame_map.get(str(raw_id))
        if value is not None:
            values.append(value)
    representative = frame_map.get(str(candidate.get("frame_id", "")))
    if representative is not None:
        values.insert(0, representative)
    return values


def kis_hits(
    candidate: dict[str, Any],
    labels: list[tuple[str, ...]],
    frame_map: dict[str, tuple[str, int, int]],
    fps_map: dict[str, float],
    tolerance_ms: float,
) -> tuple[bool, bool, bool]:
    """Return video, representative, and any-path-frame relevance flags."""

    positives = [
        (row[0], int(row[1]))
        for row in labels
        if len(row) >= 2 and row[1].strip().isdigit()
    ]
    frames = candidate_frames(candidate, frame_map)
    video = str(candidate.get("video_id", ""))
    video_hit = video in {item[0] for item in positives}

    def matches(frame: tuple[str, int, int], positive: tuple[str, int]) -> bool:
        if frame[0] != positive[0]:
            return False
        if tolerance_ms == 0:
            return frame[1] == positive[1]
        return abs(frame[2] - label_ms(positive[0], positive[1], fps_map)) <= tolerance_ms

    representative_hit = bool(frames) and any(matches(frames[0], positive) for positive in positives)
    path_hit = any(matches(frame, positive) for frame in frames for positive in positives)
    return video_hit, representative_hit, path_hit


def trake_hits(
    candidate: dict[str, Any],
    labels: list[tuple[str, ...]],
    fps_map: dict[str, float],
    tolerance_ms: float,
) -> tuple[bool, list[bool], bool]:
    """Return video, per-event, and all-event path relevance for TRAKE."""

    video = str(candidate.get("video_id", ""))
    values = [int(value) for value in candidate.get("frame_idxs", [])]
    rows = [row for row in labels if len(row) == len(values) + 1 and row[0] == video]

    def close(value: int, target: int) -> bool:
        if tolerance_ms == 0:
            return value == target
        return abs(label_ms(video, value, fps_map) - label_ms(video, target, fps_map)) <= tolerance_ms

    event_hits = [any(close(value, int(row[index + 1])) for row in rows) for index, value in enumerate(values)]
    all_hit = any(all(close(values[index], int(row[index + 1])) for index in range(len(values))) for row in rows)
    return bool(rows), event_hits, all_hit


def evaluate(response_path: Path, query_root: Path, metadata_path: Path, tolerance_seconds: float) -> dict[str, Any]:
    """Evaluate one raw-response JSON file."""

    responses = json.loads(response_path.read_text(encoding="utf-8"))
    frame_map, fps_map = load_frame_maps(metadata_path)
    tolerance_ms = tolerance_seconds * 1000.0
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for item in responses:
        query_file = str(item.get("query_file", ""))
        if item.get("status_code") != 200:
            skipped.append({"query_file": query_file, "reason": "request_failed"})
            continue
        source = Path(query_file)
        gt = query_root / source.parent.name / "ground_truth" / f"{source.stem}.csv"
        if not gt.is_file():
            skipped.append({"query_file": query_file, "reason": "no_ground_truth"})
            continue
        labels = labels_from(gt)
        body = item.get("response") or {}
        kind = str(item.get("type", "kis"))
        candidates = body.get("paths", []) if kind == "trake" else body.get("results", [])
        video_flags: list[bool] = []
        representative_flags: list[bool] = []
        path_flags: list[bool] = []
        event_matrix: list[list[bool]] = []
        for candidate in candidates:
            if kind == "trake":
                video_hit, event_hit, all_hit = trake_hits(candidate, labels, fps_map, tolerance_ms)
                video_flags.append(video_hit)
                representative_flags.append(all_hit)
                path_flags.append(all_hit)
                event_matrix.append(event_hit)
            else:
                video_hit, representative_hit, path_hit = kis_hits(candidate, labels, frame_map, fps_map, tolerance_ms)
                video_flags.append(video_hit)
                representative_flags.append(representative_hit)
                path_flags.append(path_hit)
        latency = body.get("latency", {})
        rows.append({
            "query_file": query_file,
            "type": kind,
            "labels": len(labels),
            "candidate_count": len(candidates),
            "video_first_rank": rank_of(video_flags),
            "representative_first_rank": rank_of(representative_flags),
            "path_first_rank": rank_of(path_flags),
            "video_flags": video_flags,
            "representative_flags": representative_flags,
            "path_flags": path_flags,
            "event_matrix": event_matrix,
            "latency_ms": latency.get("total_ms"),
            "retrieval_ms": latency.get("retrieval_ms"),
            "alignment_ms": latency.get("alignment_ms"),
            "wall_s": item.get("elapsed_wall_s"),
        })

    result: dict[str, Any] = {
        "response_file": str(response_path),
        "tolerance_seconds": tolerance_seconds,
        "evaluated_queries": len(rows),
        "skipped": skipped,
        "per_query": rows,
    }
    for kind in ("kis", "qa", "trake"):
        selected = [row for row in rows if row["type"] == kind]
        if not selected:
            continue
        metrics: dict[str, Any] = {"queries": len(selected)}
        for label, field in (("video", "video_flags"), ("representative", "representative_flags"), ("path", "path_flags")):
            for k in K_VALUES:
                success = sum(any(row[field][:k]) for row in selected)
                metrics[f"{label}_recall@{k}"] = success / len(selected)
                metrics[f"{label}_recall@{k}_95ci"] = wilson(success, len(selected))
            ranks = [row[f"{label}_first_rank"] for row in selected if row[f"{label}_first_rank"] is not None]
            metrics[f"{label}_mrr"] = sum(1 / rank for rank in ranks) / len(selected)
            metrics[f"{label}_median_rank"] = median(ranks) if ranks else None
        if kind == "trake":
            for k in K_VALUES:
                flags = [
                    any(candidate[event] for candidate in row["event_matrix"][:k])
                    for row in selected
                    for event in range(len(row["event_matrix"][0]) if row["event_matrix"] else 0)
                ]
                metrics[f"event_recall@{k}"] = sum(flags) / len(flags) if flags else None
        for field, label in (("latency_ms", "total"), ("retrieval_ms", "retrieval"), ("alignment_ms", "alignment")):
            values = [float(row[field]) for row in selected if row[field] is not None]
            metrics[f"{label}_latency_mean_ms"] = sum(values) / len(values) if values else None
            metrics[f"{label}_latency_p50_ms"] = percentile(values, 0.5)
            metrics[f"{label}_latency_p95_ms"] = percentile(values, 0.95)
        result[kind] = metrics
    return result


def main() -> None:
    """Write exact and relaxed evaluation JSON for one response file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    parser.add_argument("--query-root", default="artifacts/query")
    parser.add_argument("--metadata", default="artifacts/frame_store/frames.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance-seconds", type=float, default=5.0)
    args = parser.parse_args()
    result = {
        "exact": evaluate(Path(args.responses), Path(args.query_root), Path(args.metadata), 0.0),
        f"relaxed_{args.tolerance_seconds:g}s": evaluate(Path(args.responses), Path(args.query_root), Path(args.metadata), args.tolerance_seconds),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode, value in result.items():
        print(mode)
        for kind in ("kis", "qa", "trake"):
            if kind in value:
                print(kind, json.dumps(value[kind], ensure_ascii=False))


if __name__ == "__main__":
    main()
