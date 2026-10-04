#!/usr/bin/env python3
"""Execute ablation experiments comparing Static DP, Top-K, Visual Continuity, and Ours.

Verifies Phase 14 and Phase 15 of SOICT motion graph implementation plan:
- B2: Static DP (frozen baseline)
- B3: Top-K without transition edges (beta = 0)
- B4: Visual continuity (MotionCosineTransitionScorer)
- Ours: Query-conditioned transition delta (EmbeddingDeltaTransitionScorer)

Outputs machine-readable benchmark records to results/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticDataset, DiagnosticQuery
from hcmai.temporal.dp import DPPath, align_video
from hcmai.temporal.metrics import (
    MethodEvaluationSummary,
    evaluate_path_grounding,
    evaluate_retrieval_ranking,
)
from hcmai.temporal.baselines import MotionCosineTransitionScorer
from hcmai.temporal.transition_decoder import (
    DictFrameEmbeddingSource,
    EmbeddingDeltaTransitionScorer,
    FrameEmbeddingAccessor,
    FrameEmbeddingSource,
    TextEmbeddingSource,
    decode_candidate_lattice,
    decode_motion_graph_video,
    encode_query_events,
    select_event_candidates,
)


def evaluate_method_on_dataset(
    method_name: str,
    dataset: DiagnosticDataset,
    video_scores: dict[str, VideoEventScores],
    frame_embeddings: FrameEmbeddingSource | Any,
    text_encoder: TextEmbeddingSource | None = None,
    candidate_k: int = 32,
    transition_weight: float = 0.25,
    lambda_gap: float = 1e-5,
) -> MethodEvaluationSummary:
    """Evaluate one method across all queries in the diagnostic dataset.

    Ranks candidate paths decoded across all candidate videos in video_scores to compute
    true retrieval metrics (R@1, R@5, R@10, MRR) alongside grounding metrics (AllHit, EventHit).
    """
    all_hits: list[bool] = []
    event_hits_list: list[float] = []
    r1_list: list[float] = []
    r5_list: list[float] = []
    r10_list: list[float] = []
    mrr_list: list[float] = []
    latencies_ms: list[float] = []
    evaluated_edges: list[int] = []

    accessor = (
        frame_embeddings
        if isinstance(frame_embeddings, (FrameEmbeddingAccessor, DictFrameEmbeddingSource))
        else FrameEmbeddingAccessor(frame_embeddings)
    )

    for query in dataset.queries:
        if query.video_id not in video_scores:
            continue

        n_events = len(query.events)
        event_texts = [e.text for e in query.events]

        start_t = perf_counter()

        # Query event text embedding (used by proposed method)
        q_embs = None
        if method_name == "ours":
            q_embs = encode_query_events(event_texts, encoder=text_encoder)

        all_candidate_paths: list[DPPath] = []
        target_video_path: DPPath | None = None
        query_edges = 0

        # Decode all candidate videos in pool to evaluate multi-video retrieval ranking
        for vid, raw_video in video_scores.items():
            if raw_video.scores.shape[0] < n_events:
                continue
            video = VideoEventScores(
                video_id=raw_video.video_id,
                frame_ids=raw_video.frame_ids,
                frame_idx=raw_video.frame_idx,
                timestamps_ms=raw_video.timestamps_ms,
                scores=raw_video.scores[:n_events],
            )
            if method_name == "baseline_dp":
                # B2: Frozen Static DP over all frames
                paths = align_video(video, lambda_gap=lambda_gap, paths=1)
                edges_count = 0

            elif method_name == "topk_no_edges":
                # B3: Top-K candidate lattice with no transition edges (beta = 0.0)
                cand_layers = select_event_candidates(video, candidate_k=candidate_k)
                if len(cand_layers) == n_events:
                    paths = decode_candidate_lattice(
                        video,
                        candidates=cand_layers,
                        transition_scores=None,
                        transition_weight=0.0,
                        lambda_gap=lambda_gap,
                        paths=1,
                    )
                    edges_count = sum(
                        len(cand_layers[i].frame_indices) * len(cand_layers[i + 1].frame_indices)
                        for i in range(n_events - 1)
                    )
                else:
                    paths = []
                    edges_count = 0

            elif method_name == "visual_continuity":
                # B4: Visual continuity baseline (cosine similarity between consecutive frames)
                scorer = MotionCosineTransitionScorer()
                cand_layers = select_event_candidates(video, candidate_k=candidate_k)
                if cand_layers and len(cand_layers) == n_events:
                    trans_mats = []
                    for i in range(n_events - 1):
                        src_v = accessor.get_frame_embeddings(video.video_id, cand_layers[i].frame_indices)
                        tgt_v = accessor.get_frame_embeddings(video.video_id, cand_layers[i + 1].frame_indices)
                        src_t = video.timestamps_ms[cand_layers[i].frame_indices]
                        tgt_t = video.timestamps_ms[cand_layers[i + 1].frame_indices]
                        mat = scorer.compute_transition_matrix(src_v, tgt_v, src_t, tgt_t)
                        trans_mats.append(mat)
                    paths = decode_candidate_lattice(
                        video,
                        candidates=cand_layers,
                        transition_scores=trans_mats,
                        transition_weight=transition_weight,
                        lambda_gap=lambda_gap,
                        paths=1,
                    )
                    edges_count = sum(
                        len(cand_layers[i].frame_indices) * len(cand_layers[i + 1].frame_indices)
                        for i in range(n_events - 1)
                    )
                else:
                    paths = []
                    edges_count = 0

            elif method_name == "ours":
                # Proposed: Query-conditioned signed transition graph
                assert q_embs is not None
                paths = decode_motion_graph_video(
                    video,
                    frame_embeddings=accessor,
                    event_embeddings=q_embs,
                    candidate_k=candidate_k,
                    transition_weight=transition_weight,
                    lambda_gap=lambda_gap,
                    paths=1,
                )
                cand_layers = select_event_candidates(video, candidate_k=candidate_k)
                edges_count = sum(
                    len(cand_layers[i].frame_indices) * len(cand_layers[i + 1].frame_indices)
                    for i in range(n_events - 1)
                ) if len(cand_layers) == n_events else 0
            else:
                raise ValueError(f"Unknown method name: {method_name}")

            query_edges += edges_count
            if paths:
                all_candidate_paths.append(paths[0])
                if vid == query.video_id:
                    target_video_path = paths[0]

        latency = (perf_counter() - start_t) * 1000.0
        latencies_ms.append(latency)
        evaluated_edges.append(query_edges // max(len(video_scores), 1))

        # 1. Evaluate Grounding on target video path
        if target_video_path is not None:
            grounding = evaluate_path_grounding(target_video_path, query, video_scores[query.video_id])
            all_hits.append(grounding.all_hit)
            event_hits_list.append(grounding.hit_rate)
        else:
            all_hits.append(False)
            event_hits_list.append(0.0)

        # 2. Evaluate Multi-Video Retrieval Ranking across all candidate videos
        ranked_paths = sorted(all_candidate_paths, key=lambda p: p.score, reverse=True)
        ret = evaluate_retrieval_ranking(ranked_paths, query.video_id)
        r1_list.append(ret.r1)
        r5_list.append(ret.r5)
        r10_list.append(ret.r10)
        mrr_list.append(ret.mrr)

    n_q = len(all_hits)
    return MethodEvaluationSummary(
        method_name=method_name,
        num_queries=n_q,
        r1=float(np.mean(r1_list)) if r1_list else 0.0,
        r5=float(np.mean(r5_list)) if r5_list else 0.0,
        r10=float(np.mean(r10_list)) if r10_list else 0.0,
        mrr=float(np.mean(mrr_list)) if mrr_list else 0.0,
        event_hit=float(np.mean(event_hits_list)) if event_hits_list else 0.0,
        all_hit=float(np.mean(all_hits)) if all_hits else 0.0,
        avg_edges_per_video=float(np.mean(evaluated_edges)) if evaluated_edges else 0.0,
        avg_latency_ms=float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    )


def save_summary_result(summary: MethodEvaluationSummary, output_path: Path | str) -> None:
    """Save evaluation summary as machine-readable JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run motion graph ablation experiments")
    parser.add_argument(
        "--queries",
        type=str,
        default="data/eval/transition_diagnostic.json",
        help="Path to diagnostic dataset JSON",
    )
    parser.add_argument("--candidate-k", type=int, default=32, help="Candidate count K")
    parser.add_argument("--transition-weight", type=float, default=0.25, help="Transition weight beta")
    parser.add_argument("--lambda-gap", type=float, default=1e-5, help="Time gap penalty lambda")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON results")
    parser.add_argument("--use-real-siglip", action="store_true", help="Instantiate real SigLIP adapter")
    parser.add_argument("--device", type=str, default="cpu", help="Device for encoder")
    args = parser.parse_args()

    query_path = Path(args.queries)
    if not query_path.is_file():
        print(f"Queries file {query_path} not found.")
        return

    dataset = DiagnosticDataset.load_json(query_path)
    print(f"Loaded {len(dataset)} diagnostic queries from {query_path}.")

    # Configure text encoder
    encoder: TextEmbeddingSource
    if args.use_real_siglip:
        from hcmai.retrieval.embedding.adapters.siglip import SigLIPAdapter
        from hcmai.common.config import EncoderConfig
        print("Initializing real SigLIP text encoder...")
        encoder = SigLIPAdapter(EncoderConfig(model_name="google/siglip2-base-patch16-224", device=args.device))
    else:
        from scripts.evaluation.eval_corpus_builder import DeterministicTextEncoder
        encoder = DeterministicTextEncoder(dim=64)

    # Build evaluation corpus (targets + distractors)
    from scripts.evaluation.eval_corpus_builder import build_synthetic_eval_corpus
    video_scores, frame_embeddings = build_synthetic_eval_corpus(dataset, text_encoder=encoder)
    print(f"Prepared benchmark corpus with {len(video_scores)} candidate videos.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["baseline_dp", "topk_no_edges", "visual_continuity", "ours"]
    print("\n| Method | R@1 | R@5 | MRR | EventHit | AllHit | Avg. Edges | Latency |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|")

    for method in methods:
        summary = evaluate_method_on_dataset(
            method_name=method,
            dataset=dataset,
            video_scores=video_scores,
            frame_embeddings=frame_embeddings,
            text_encoder=encoder,
            candidate_k=args.candidate_k,
            transition_weight=args.transition_weight,
            lambda_gap=args.lambda_gap,
        )
        print(summary.to_table_row())
        save_summary_result(summary, out_dir / f"{method}.json")

    print(f"\nAll ablation results saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
