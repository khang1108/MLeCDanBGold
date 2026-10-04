"""Run retrieval once per labelled query and freeze the target video's score matrix.

Stores ``<query_id>.npz`` (frame_ids, frame_idx, timestamps_ms, scores, gt_ms) plus one
``meta.jsonl`` row with retrieval latency and the video-level first-pass rank. Event
texts come from the labels file, so the query planner never runs here.

Usage:
    PYTHONPATH=src:. aic/bin/python -m scripts.evaluation.dump_event_scores \\
        --labels benchmark/event_labels_v1.yaml --out-dir runs/correction-eval-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from hcmai.orchestration.setup import load_search_service
from hcmai.retrieval.plan import KISRetrievalEvent, KISRetrievalPlan
from hcmai.retrieval.retriever.video_scores import VideoEventScores


def gt_interval(event: dict, video: VideoEventScores, tolerance_ms: int) -> list[int]:
    """Normalize either ``interval_ms`` or an organizer ``frame_idx`` into a closed ms interval."""
    if "interval_ms" in event:
        return [int(event["interval_ms"][0]), int(event["interval_ms"][1])]
    pos = int(np.argmin(np.abs(video.frame_idx.astype(np.int64) - int(event["frame_idx"]))))
    t = int(video.timestamps_ms[pos])
    return [t - tolerance_ms, t + tolerance_ms]


def main() -> int:
    """Dump frozen matrices for every labelled query."""
    ap = argparse.ArgumentParser(description="Freeze per-query score matrices for the correction study")
    ap.add_argument("--labels", type=Path, default=Path("benchmark/event_labels_v1.yaml"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tolerance-ms", type=int, default=2_000)
    ap.add_argument("--no-bm25", action="store_true", help="Disable BM25 components")
    ap.add_argument("--rank-depth", type=int, default=500, help="Path rows searched to find the target's rank")
    args = ap.parse_args()

    messages: list[str] = []
    search = load_search_service(messages).temporal
    if search is None:
        print(f"temporal search unavailable: {messages}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_bm25 = not args.no_bm25

    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    with open(args.out_dir / "meta.jsonl", "w", encoding="utf-8") as meta:
        for item in labels:
            events = [e["text"] for e in item["events"]]
            # Event texts come straight from the labels, so no planner or LLM rewrite runs.
            plan = KISRetrievalPlan(events=tuple(
                KISRetrievalEvent(f"E{i}", canonical_text=t, dense_text=t, bm25_text=t)
                for i, t in enumerate(events, 1)
            ))

            # Video-level rank: deduplicate path rows by video, in order.
            result = search.search_plan(plan, use_dense=True, use_bm25=use_bm25, top_k=args.rank_depth)
            order = list(dict.fromkeys(path.video_id for path in result.paths))
            rank = order.index(item["video_id"]) + 1 if item["video_id"] in order else None

            scored = search.score_video(plan, video_id=item["video_id"], use_dense=True, use_bm25=use_bm25)
            target, config = scored.video, scored.decoder_config
            retrieval_ms = result.retrieval_ms

            gt = [gt_interval(e, target, args.tolerance_ms) for e in item["events"]]
            np.savez(
                args.out_dir / f"{item['query_id']}.npz",
                frame_ids=target.frame_ids.astype(str),
                frame_idx=target.frame_idx,
                timestamps_ms=target.timestamps_ms,
                scores=target.scores,
                gt_ms=np.array(gt, dtype=np.int64),
            )
            meta.write(json.dumps({
                "query_id": item["query_id"],
                "subset": item["subset"],
                "video_id": item["video_id"],
                "n_events": len(events),
                "video_rank": rank,
                "retrieval_ms": round(retrieval_ms, 1),
                "lambda_gap": config.lambda_gap,
                "use_bm25": use_bm25,
            }) + "\n")
            print(f"{item['query_id']}: rank {rank}, {len(events)} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
