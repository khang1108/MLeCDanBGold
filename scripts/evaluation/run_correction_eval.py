"""Replay every correction condition on frozen score matrices and print Hit@B by subset.

Usage:
    PYTHONPATH=src:. aic/bin/python -m scripts.evaluation.run_correction_eval \\
        --dump-dir runs/correction-eval-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.dp import align_video
from scripts.evaluation.correction_sim import SimConfig, auroc, event_margins, simulate

PROPOSALS = ("local", "path")
ORDERS = ("sequential", "random", "margin")
PROPAGATE = (True, False)


def load_dump(path: Path) -> tuple[VideoEventScores, list[tuple[int, int]]]:
    """Load one frozen matrix and its per-event ground-truth intervals."""
    z = np.load(path, allow_pickle=False)
    video = VideoEventScores(path.stem, z["frame_ids"], z["frame_idx"], z["timestamps_ms"], z["scores"])
    return video, [(int(a), int(b)) for a, b in z["gt_ms"]]


def margin_diagnostics(data: dict, lambda_gap: float, separation_ms: int, tolerance_ms: int) -> tuple[float, int, int, float]:
    """AUROC of first-decode margins against wrong events, plus median margin time in ms."""
    wrong: list[float] = []
    right: list[float] = []
    elapsed: list[float] = []
    for video, gt in data.values():
        paths = align_video(video, lambda_gap, 1)
        if not paths:
            continue
        position = {str(f): i for i, f in enumerate(video.frame_ids)}
        start = perf_counter()
        margins = event_margins(video, lambda_gap=lambda_gap, min_separation_ms=separation_ms)
        elapsed.append((perf_counter() - start) * 1_000)
        for event, fid in enumerate(paths[0].frame_ids):
            t = int(video.timestamps_ms[position[fid]])
            (right if gt[event][0] - tolerance_ms <= t <= gt[event][1] + tolerance_ms else wrong).append(float(margins[event]))
    return auroc(wrong, right), len(wrong), len(right), float(np.median(elapsed)) if elapsed else float("nan")


def main() -> int:
    """Sweep proposal source x inspection order and report action counts."""
    ap = argparse.ArgumentParser(description="Replay correction conditions on frozen score matrices")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--separation-ms", type=int, default=5_000)
    ap.add_argument("--tolerance-ms", type=int, default=1_000, help="Ground-truth interval slack on both sides")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lambda-gap", type=float, default=1e-5)
    args = ap.parse_args()

    meta = {
        row["query_id"]: row
        for row in map(json.loads, (args.dump_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines())
    }
    data = {q: load_dump(args.dump_dir / f"{q}.npz") for q in meta}

    rows = []
    conditions = [(p, o, g) for g in PROPAGATE for p in PROPOSALS for o in ORDERS]
    for proposals, order, propagate in conditions:
            for q, (video, gt) in data.items():
                for seed in range(args.seeds if order == "random" else 1):
                    cfg = SimConfig(
                        proposals=proposals, order=order, k=args.k,
                        separation_ms=args.separation_ms, tolerance_ms=args.tolerance_ms,
                        budget=args.budget, propagate=propagate, lambda_gap=args.lambda_gap, seed=seed,
                    )
                    rows.append(simulate(video, gt, cfg) | {
                        "query_id": q, "subset": meta[q]["subset"],
                        "proposals": proposals, "order": order, "propagate": propagate, "seed": seed,
                    })

    out_path = args.dump_dir / f"results_k{args.k}_sep{args.separation_ms}_tol{args.tolerance_ms}.jsonl"
    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    score, n_wrong, n_right, margin_ms = margin_diagnostics(data, args.lambda_gap, args.separation_ms, args.tolerance_ms)
    retrieval_ms = float(np.median([m["retrieval_ms"] for m in meta.values()]))
    print(f"queries={len(data)}  k={args.k}  separation_ms={args.separation_ms}  tolerance_ms={args.tolerance_ms}  budget={args.budget}")
    print(f"AUROC(margin -> wrong event) = {score:.3f}  (wrong events={n_wrong}, correct events={n_right})")
    print(f"median margin computation = {margin_ms:.1f} ms; median retrieval = {retrieval_ms:.1f} ms")

    subsets = sorted({m["subset"] for m in meta.values()}) + ["all"]
    for subset in subsets:
        print(f"\n== {subset} ==")
        for proposals, order, propagate in conditions:
                sel = [r for r in rows if r["proposals"] == proposals and r["order"] == order
                       and r["propagate"] == propagate and (subset == "all" or r["subset"] == subset)]
                hits = [np.mean([r["solved"] and r["actions"] <= b for r in sel]) for b in range(args.budget + 1)]
                solved = [r["actions"] for r in sel if r["solved"]]
                mean_actions = f"{np.mean(solved):.2f}" if solved else "  - "
                fixed = sum(r["auto_fixed"] for r in sel) / len(sel)
                broken = sum(r["auto_broken"] for r in sel) / len(sel)
                mode = "joint " if propagate else "frozen"
                print(f"{mode} {proposals:5s} {order:10s} n={len(sel):3d} actions={mean_actions} "
                      + " ".join(f"H@{b}={h:.2f}" for b, h in enumerate(hits))
                      + f"  auto+={fixed:.2f} auto-={broken:.2f}")
    print(f"\nrows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
