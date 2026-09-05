"""Score ablation output against benchmark labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KS = (1, 5, 10, 20, 50)


def _load(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _hit_ranks(label: dict[str, Any], paths: list[dict[str, Any]], tolerance_ms: int) -> tuple:
    """Return the 1-based video-rank and path-rank of the first hits.

    ``video_rank`` counts distinct videos, not paths. With ``paths_per_video``
    above one a single video occupies several consecutive rows, so ranking by
    path position reports a video as up to ``paths_per_video`` times worse than
    the operator sees it.
    """

    gold: dict[str, list[int]] = {}
    for candidate in label["candidates"]:
        gold.setdefault(str(candidate["video_id"]), []).append(int(candidate["timestamp_ms"]))
    video_rank: int | None = None
    frame_rank: int | None = None
    seen: dict[str, int] = {}
    for rank, path in enumerate(paths, start=1):
        video_id = str(path["video_id"])
        position = seen.setdefault(video_id, len(seen) + 1)
        golds = gold.get(video_id)
        if golds is None:
            continue
        if video_rank is None:
            video_rank = position
        if any(
            abs(gold_ms - int(value)) <= tolerance_ms
            for gold_ms in golds
            for value in path["timestamps_ms"]
        ):
            frame_rank = rank
            break
    return video_rank, frame_rank


def summarize(
    labels: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    tolerance_ms: int,
    sources: set[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate Recall@K and MRR for every ablation run in the output file."""

    report: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = labels.get(str(row["query_id"]))
        if label is None or label["task"] == "trake" or str(label["source"]) not in sources:
            continue
        run = report.setdefault(
            str(row["run"]),
            {
                "n": 0,
                "video": dict.fromkeys(KS, 0),
                "frame": dict.fromkeys(KS, 0),
                "rr": 0.0,
                "frr": 0.0,
            },
        )
        video_rank, frame_rank = _hit_ranks(label, row.get("paths", []), tolerance_ms)
        run["n"] += 1
        if video_rank is not None:
            run["rr"] += 1.0 / video_rank
            for k in KS:
                run["video"][k] += int(video_rank <= k)
        if frame_rank is not None:
            run["frr"] += 1.0 / frame_rank
            for k in KS:
                run["frame"][k] += int(frame_rank <= k)
    for run in report.values():
        run["mrr"] = round(run["rr"] / run["n"], 4) if run["n"] else 0.0
        run["frame_mrr"] = round(run["frr"] / run["n"], 4) if run["n"] else 0.0
    return report


def main() -> None:
    """Print a Recall@K and MRR table per ablation run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("benchmark/labels.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/p0_ablation_results.jsonl"))
    parser.add_argument("--tolerance-ms", type=int, default=2000)
    parser.add_argument("--sources", nargs="+", default=["curated", "top10"])
    arguments = parser.parse_args()
    labels ={str(record["query_id"]): record for record in _load(arguments.labels)}
    rows = _load(arguments.results)
    report = summarize(labels, rows, arguments.tolerance_ms, set(arguments.sources))

    header = "".join(f"  V@{k:<5}" for k in KS) + "".join(f"  F@{k:<5}" for k in KS)
    print(f"{'run':<24}{'n':>4}{header}   V-MRR  F-MRR")
    for name in sorted(report):
        run = report[name]
        video = "".join(f"  {run['video'][k]:>2}/{run['n']:<3}" for k in KS)
        frame = "".join(f"  {run['frame'][k]:>2}/{run['n']:<3}" for k in KS)
        print(f"{name:<24}{run['n']:>4}{video}{frame}   {run['mrr']:.3f}  {run['frame_mrr']:.3f}")


if __name__ == "__main__":
    main()
