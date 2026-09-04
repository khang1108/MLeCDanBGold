#!/usr/bin/env python3
"""CLI utility to inspect component-level temporal evidence and calibration.

Usage:
    PYTHONPATH=src:. aic/bin/python scripts/debug_temporal_evidence.py \\
        --query-file tests/fixtures/l26_v254_query.yaml \\
        --run B3 \\
        --video-id L26_V254 \\
        --top-frames 12 \\
        --use-bm25
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np
import yaml

from hcmai.orchestration.setup import load_search_service
from hcmai.retrieval.evidence.ablation import make_ablation_scorer
from hcmai.retrieval.evidence.hybrid import TemporalEvidenceScorer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic arguments, including the B-series scoring stage."""

    parser = argparse.ArgumentParser(
        description="Debug component-level temporal evidence for a query"
    )
    parser.add_argument(
        "--run",
        default="B5",
        help="Adaptive B-series run to inspect (B1-B6; default: B5)",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        required=True,
        help="Path to YAML file containing query and retrieval_events",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        default=[],
        help="Target video ID to filter frame positions (repeatable)",
    )
    parser.add_argument(
        "--top-frames",
        type=int,
        default=10,
        help="Number of top frames to display (default: 10)",
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
    return parser.parse_args(argv)


def _configure_debug_run(
    baseline: TemporalEvidenceScorer,
    run_key: str,
    *,
    use_dense: bool,
    use_bm25: bool,
) -> tuple[TemporalEvidenceScorer, bool, bool]:
    """Create an isolated adaptive run and combine run/source CLI switches."""

    scorer, run_defaults = make_ablation_scorer(baseline, run_key)
    if scorer.config.fusion_mode != "adaptive_p0":
        raise ValueError(
            "component diagnostics require an adaptive run (choose B1-B6, not B0)"
        )
    return (
        scorer,
        use_dense and bool(run_defaults["use_dense"]),
        use_bm25 and bool(run_defaults["use_bm25"]),
    )


def main() -> int:
    args = parse_args()

    if not args.query_file.is_file():
        print(f"Error: query file not found: {args.query_file}", file=sys.stderr)
        return 1

    with open(args.query_file, encoding="utf-8") as f:
        query_data = yaml.safe_load(f)

    raw_query = query_data.get("query", "")
    if isinstance(raw_query, str):
        original_events = [
            line.strip()
            for line in raw_query.strip().splitlines()
            if line.strip()
        ]
    elif isinstance(raw_query, list):
        original_events = [str(x).strip() for x in raw_query if str(x).strip()]
    else:
        original_events = []

    retrieval_events = query_data.get("retrieval_events", [])
    caption_events = query_data.get("caption_events", original_events)

    if not original_events or len(original_events) != len(retrieval_events):
        print(
            f"Error: event count mismatch (original: {len(original_events)}, "
            f"retrieval: {len(retrieval_events)})",
            file=sys.stderr,
        )
        return 1

    messages: list[str] = []
    service = load_search_service(messages)
    for msg in messages:
        print(f"[{msg}]", file=sys.stderr)

    temporal = getattr(service, "temporal_evidence", None)
    if not isinstance(temporal, TemporalEvidenceScorer):
        print(
            "Error: SearchService.temporal_evidence is not a TemporalEvidenceScorer instance",
            file=sys.stderr,
        )
        return 1

    try:
        run_scorer, run_use_dense, run_use_bm25 = _configure_debug_run(
            temporal,
            args.run,
            use_dense=args.use_dense,
            use_bm25=args.use_bm25,
        )
    except (KeyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    visual_index = run_scorer.visual_index
    run_label = args.run.upper()

    print(f"\nExecuting {run_label} debug scoring over corpus...")
    bundle, calibrated, fused = run_scorer._prepare_adaptive_components(
        original_events=original_events,
        retrieval_events=retrieval_events,
        caption_events=caption_events,
        use_dense=run_use_dense,
        use_bm25=run_use_bm25,
    )

    target_video_ids = args.video_ids or sorted(
        {str(video_id) for video_id in visual_index.video_ids}
    )

    for video_id in target_video_ids:
        positions = visual_index.video_positions(video_id)
        if len(positions) == 0:
            print(f"\nWarning: video {video_id} has no frames in index.", file=sys.stderr)
            continue

        print(
            f"\n================ Video: {video_id} "
            f"({len(positions)} frames) ================"
        )
        header = (
            f"{'Event':<6} {'Component':<15} {'RawMax':<8} {'RawMed':<8} "
            f"{'CalMax':<8} {'Rel':<6} {'Cov':<6} {'TopFrameIdx'}"
        )
        print(header)
        print("-" * 95)

        for event_idx, (orig, ret) in enumerate(
            zip(original_events, retrieval_events, strict=True)
        ):
            print(f"E{event_idx + 1}: {orig[:45]}... -> {ret[:45]}...")
            router_weights = run_scorer._adaptive_scorer().router.multipliers(
                orig, ret
            )
            weights_str = ", ".join(
                f"{name}: {router_weights.get(name, 0.0):.3f}"
                for name in bundle.components
            )
            print(f"    [Requested component weights]: {weights_str}")

            for name, comp in bundle.components.items():
                raw_in_vid = comp.raw_scores[event_idx, positions]
                cal_in_vid = calibrated[name].scores[event_idx, positions]

                raw_max = float(raw_in_vid.max()) if len(raw_in_vid) else 0.0
                cal_max = float(cal_in_vid.max()) if len(cal_in_vid) else 0.0
                reliability = float(calibrated[name].reliability[event_idx])

                if comp.coverage is not None:
                    cov_in_vid = comp.coverage[positions]
                    coverage_pct = f"{int(np.mean(cov_in_vid.astype(np.float32)) * 100)}%"
                    supported_in_vid = cov_in_vid
                elif name.startswith("bm25_"):
                    coverage_pct = "100%"
                    supported_in_vid = raw_in_vid > 0.0
                else:
                    coverage_pct = "100%"
                    supported_in_vid = None

                if supported_in_vid is not None and np.any(supported_in_vid):
                    raw_med = float(np.median(raw_in_vid[supported_in_vid]))
                else:
                    raw_med = float(np.median(raw_in_vid)) if len(raw_in_vid) else 0.0

                top_k = min(args.top_frames, len(cal_in_vid))
                if top_k > 0:
                    top_pos = np.argsort(-cal_in_vid, kind="stable")[:top_k]
                    top_frame_indices = visual_index.frame_idx[positions[top_pos]]
                    top_frames_str = ",".join(str(idx) for idx in top_frame_indices)
                else:
                    top_frames_str = "none"

                print(
                    f"E{event_idx + 1:<5} {name:<15} {raw_max:<8.3f} {raw_med:<8.3f} {cal_max:<8.3f} {reliability:<6.2f} {coverage_pct:<6} {top_frames_str}"
                )

            # Print fused score row for this event
            fused_in_vid = fused[event_idx, positions]
            fused_max = float(fused_in_vid.max()) if len(fused_in_vid) else 0.0
            fused_med = float(np.median(fused_in_vid)) if len(fused_in_vid) else 0.0
            top_k = min(args.top_frames, len(fused_in_vid))
            top_pos = np.argsort(-fused_in_vid, kind="stable")[:top_k]
            fused_frames = visual_index.frame_idx[positions[top_pos]]
            fused_frames_str = ",".join(str(idx) for idx in fused_frames)
            print(
                f"E{event_idx + 1:<5} {('fused_' + run_label):<15} "
                f"{fused_max:<8.3f} {fused_med:<8.3f} {fused_max:<8.3f} "
                f"{'1.00':<6} {'100%':<6} {fused_frames_str}"
            )
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
