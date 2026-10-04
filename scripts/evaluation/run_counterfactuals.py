#!/usr/bin/env python3
"""Run reverse and shuffled counterfactual experiments for motion graph decoding.

Verifies Phase 10 of SOICT motion graph implementation plan:
- Evaluates forward query score S(Q, V) vs reversed query score S(Q_rev, V).
- Evaluates original transition edges vs permuted/shuffled transition edges.
- Saves machine-readable results to results/reverse.json and results/shuffled.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticDataset, DiagnosticQuery
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


@dataclass(frozen=True, slots=True)
class QueryCounterfactualResult:
    query_id: str
    video_id: str
    original_score: float
    counterfactual_score: float
    margin: float
    original_preferred: bool


def evaluate_reverse_counterfactuals(
    dataset: DiagnosticDataset,
    video_scores: dict[str, VideoEventScores],
    frame_embeddings: FrameEmbeddingSource | Any,
    text_encoder: TextEmbeddingSource | None = None,
    candidate_k: int = 32,
    transition_weight: float = 0.25,
    lambda_gap: float = 1e-5,
) -> dict[str, Any]:
    """Evaluate forward queries against chronological reversed queries."""
    results: list[QueryCounterfactualResult] = []

    for query in dataset.queries:
        if query.video_id not in video_scores:
            continue
        raw_video = video_scores[query.video_id]
        n_events = len(query.events)
        if raw_video.scores.shape[0] != n_events:
            video = VideoEventScores(
                video_id=raw_video.video_id,
                frame_ids=raw_video.frame_ids,
                frame_idx=raw_video.frame_idx,
                timestamps_ms=raw_video.timestamps_ms,
                scores=raw_video.scores[:n_events],
            )
        else:
            video = raw_video

        # 1. Forward query
        fwd_texts = [e.text for e in query.events]
        fwd_embs = encode_query_events(fwd_texts, encoder=text_encoder)
        fwd_paths = decode_motion_graph_video(
            video,
            frame_embeddings=frame_embeddings,
            event_embeddings=fwd_embs,
            candidate_k=candidate_k,
            transition_weight=transition_weight,
            lambda_gap=lambda_gap,
            paths=1,
        )
        fwd_score = fwd_paths[0].score if fwd_paths else -np.inf

        # 2. Reversed query
        rev_query = query.reversed_query()
        rev_texts = [e.text for e in rev_query.events]
        rev_embs = encode_query_events(rev_texts, encoder=text_encoder)

        # Reversed video scores: invert event rows
        rev_scores_mat = video.scores[::-1].copy()
        rev_video = VideoEventScores(
            video_id=video.video_id,
            frame_ids=video.frame_ids,
            frame_idx=video.frame_idx,
            timestamps_ms=video.timestamps_ms,
            scores=rev_scores_mat,
        )

        rev_paths = decode_motion_graph_video(
            rev_video,
            frame_embeddings=frame_embeddings,
            event_embeddings=rev_embs,
            candidate_k=candidate_k,
            transition_weight=transition_weight,
            lambda_gap=lambda_gap,
            paths=1,
        )
        rev_score = rev_paths[0].score if rev_paths else -np.inf

        margin = float(fwd_score - rev_score) if np.isfinite(fwd_score) and np.isfinite(rev_score) else 0.0
        preferred = fwd_score > rev_score

        results.append(
            QueryCounterfactualResult(
                query_id=query.query_id,
                video_id=query.video_id,
                original_score=float(fwd_score) if np.isfinite(fwd_score) else 0.0,
                counterfactual_score=float(rev_score) if np.isfinite(rev_score) else 0.0,
                margin=margin,
                original_preferred=preferred,
            )
        )

    acc = float(np.mean([r.original_preferred for r in results])) if results else 0.0
    mean_margin = float(np.mean([r.margin for r in results])) if results else 0.0
    mean_fwd = float(np.mean([r.original_score for r in results])) if results else 0.0
    mean_rev = float(np.mean([r.counterfactual_score for r in results])) if results else 0.0

    return {
        "experiment": "reverse_counterfactual",
        "num_queries": len(results),
        "reverse_accuracy": acc,
        "mean_forward_score": mean_fwd,
        "mean_reverse_score": mean_rev,
        "mean_margin": mean_margin,
        "per_query_results": [asdict(r) for r in results],
    }


def evaluate_shuffled_counterfactuals(
    dataset: DiagnosticDataset,
    video_scores: dict[str, VideoEventScores],
    frame_embeddings: FrameEmbeddingSource | Any,
    text_encoder: TextEmbeddingSource | None = None,
    candidate_k: int = 32,
    transition_weight: float = 0.25,
    lambda_gap: float = 1e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate original transition matrices against randomly shuffled edge matrices."""
    rng = np.random.default_rng(seed)
    results: list[QueryCounterfactualResult] = []

    accessor = (
        frame_embeddings
        if isinstance(frame_embeddings, (FrameEmbeddingAccessor, DictFrameEmbeddingSource))
        else FrameEmbeddingAccessor(frame_embeddings)
    )
    scorer = EmbeddingDeltaTransitionScorer()

    for query in dataset.queries:
        if query.video_id not in video_scores:
            continue
        raw_video = video_scores[query.video_id]
        n_events = len(query.events)
        if raw_video.scores.shape[0] != n_events:
            video = VideoEventScores(
                video_id=raw_video.video_id,
                frame_ids=raw_video.frame_ids,
                frame_idx=raw_video.frame_idx,
                timestamps_ms=raw_video.timestamps_ms,
                scores=raw_video.scores[:n_events],
            )
        else:
            video = raw_video
        event_texts = [e.text for e in query.events]
        q_embs = encode_query_events(event_texts, encoder=text_encoder)

        # 1. Original motion graph decode
        orig_paths = decode_motion_graph_video(
            video,
            frame_embeddings=accessor,
            event_embeddings=q_embs,
            candidate_k=candidate_k,
            transition_weight=transition_weight,
            lambda_gap=lambda_gap,
            paths=1,
        )
        orig_score = orig_paths[0].score if orig_paths else -np.inf

        # 2. Shuffled transition decode: keep candidates fixed, randomly permute edges
        cand_layers = select_event_candidates(video, candidate_k=candidate_k)
        if len(cand_layers) == n_events:
            cand_embs = [accessor.get_frame_embeddings(video.video_id, layer.frame_indices) for layer in cand_layers]
            orig_trans = scorer.score_all_transitions(cand_embs, q_embs)

            # Shuffle rows/cols of each transition matrix
            shuffled_trans = []
            for mat in orig_trans:
                shuffled = mat.copy()
                rng.shuffle(shuffled)  # shuffle rows
                shuffled = shuffled[:, rng.permutation(shuffled.shape[1])]  # shuffle columns
                shuffled_trans.append(shuffled)

            shuff_paths = decode_candidate_lattice(
                video,
                candidates=cand_layers,
                transition_scores=shuffled_trans,
                transition_weight=transition_weight,
                lambda_gap=lambda_gap,
                paths=1,
            )
            shuff_score = shuff_paths[0].score if shuff_paths else -np.inf
        else:
            shuff_score = -np.inf

        margin = float(orig_score - shuff_score) if np.isfinite(orig_score) and np.isfinite(shuff_score) else 0.0
        preferred = orig_score >= shuff_score

        results.append(
            QueryCounterfactualResult(
                query_id=query.query_id,
                video_id=query.video_id,
                original_score=float(orig_score) if np.isfinite(orig_score) else 0.0,
                counterfactual_score=float(shuff_score) if np.isfinite(shuff_score) else 0.0,
                margin=margin,
                original_preferred=preferred,
            )
        )

    degradation_rate = float(np.mean([r.original_preferred for r in results])) if results else 0.0
    mean_margin = float(np.mean([r.margin for r in results])) if results else 0.0
    mean_orig = float(np.mean([r.original_score for r in results])) if results else 0.0
    mean_shuff = float(np.mean([r.counterfactual_score for r in results])) if results else 0.0

    return {
        "experiment": "shuffled_edge_counterfactual",
        "num_queries": len(results),
        "shuffled_degradation_rate": degradation_rate,
        "mean_original_score": mean_orig,
        "mean_shuffled_score": mean_shuff,
        "mean_margin": mean_margin,
        "per_query_results": [asdict(r) for r in results],
    }


def save_counterfactual_result(data: dict[str, Any], output_path: Path | str) -> None:
    """Save counterfactual result as machine-readable JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reverse and shuffled counterfactual experiments")
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

    # Build evaluation corpus
    from scripts.evaluation.eval_corpus_builder import build_synthetic_eval_corpus
    video_scores, frame_embeddings = build_synthetic_eval_corpus(dataset, text_encoder=encoder)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nExecuting Reverse Counterfactual Experiment...")
    rev_res = evaluate_reverse_counterfactuals(
        dataset=dataset,
        video_scores=video_scores,
        frame_embeddings=frame_embeddings,
        text_encoder=encoder,
        candidate_k=args.candidate_k,
        transition_weight=args.transition_weight,
        lambda_gap=args.lambda_gap,
    )
    save_counterfactual_result(rev_res, out_dir / "reverse.json")
    print(f"Reverse Accuracy: {rev_res['reverse_accuracy']*100:.1f}%, Mean Margin: {rev_res['mean_margin']:.4f}")

    print("\nExecuting Shuffled Edge Counterfactual Experiment...")
    shuff_res = evaluate_shuffled_counterfactuals(
        dataset=dataset,
        video_scores=video_scores,
        frame_embeddings=frame_embeddings,
        text_encoder=encoder,
        candidate_k=args.candidate_k,
        transition_weight=args.transition_weight,
        lambda_gap=args.lambda_gap,
    )
    save_counterfactual_result(shuff_res, out_dir / "shuffled.json")
    print(f"Degradation Rate: {shuff_res['shuffled_degradation_rate']*100:.1f}%, Mean Margin: {shuff_res['mean_margin']:.4f}")

    print(f"\nCounterfactual results saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
