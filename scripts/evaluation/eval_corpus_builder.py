"""Synthetic and precomputed evaluation corpus builder for SOICT motion graph evaluation.

Provides deterministic test corpora and embedding sources for running
ablation benchmarks, candidate recall evaluation, and counterfactual tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from hcmai.retrieval.retriever.video_scores import VideoEventScores
from hcmai.temporal.diagnostic import DiagnosticDataset
from hcmai.temporal.transition_decoder import TextEmbeddingSource


class DeterministicTextEncoder:
    """Deterministic hash-based text encoder conforming to TextEmbeddingSource.

    Produces reproducible L2-normalized D-dimensional vectors for query strings.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for text in texts:
            # Deterministic hash seed
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            vectors.append(vec / max(norm, 1e-12))
        return np.vstack(vectors)


def build_synthetic_eval_corpus(
    dataset: DiagnosticDataset,
    text_encoder: TextEmbeddingSource | None = None,
    num_frames: int = 80,
    dim: int = 64,
    num_distractors: int = 5,
    seed: int = 42,
) -> tuple[dict[str, VideoEventScores], dict[str, np.ndarray]]:
    """Build a multi-video evaluation corpus containing ground truth targets and distractors.

    Args:
        dataset: DiagnosticDataset containing multi-event queries.
        text_encoder: Text encoder conforming to TextEmbeddingSource.
        num_frames: Number of frames per video.
        dim: Embedding dimension.
        num_distractors: Number of distractor videos to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (video_scores_map, frame_embeddings_map).
    """
    encoder = text_encoder if text_encoder is not None else DeterministicTextEncoder(dim=dim)
    rng = np.random.default_rng(seed)

    video_scores: dict[str, VideoEventScores] = {}
    frame_embeddings: dict[str, np.ndarray] = {}

    # Compute max events across queries
    max_events = max(len(q.events) for q in dataset.queries) if dataset.queries else 2

    # 1. Build target videos for each query
    for query in dataset.queries:
        vid = query.video_id
        n_events = len(query.events)
        event_texts = [e.text for e in query.events]
        q_embs = encoder.encode_text(event_texts)

        # Baseline noise unary scores in [0.05, 0.20] with shape (max_events, num_frames)
        scores = rng.uniform(0.05, 0.20, size=(max_events, num_frames)).astype(np.float32)
        timestamps = np.arange(num_frames, dtype=np.int64) * 100
        frame_idx = np.arange(num_frames, dtype=np.int64)
        frame_ids = np.array([f"{vid}_f{i:03d}" for i in range(num_frames)])

        # Frame visual embeddings initialized with mild noise
        embs = rng.normal(0.0, 0.1, size=(num_frames, dim)).astype(np.float32)

        for e_idx, event in enumerate(query.events):
            target_idxs = list(event.target_frame_idxs)
            if not target_idxs and event.start_ms is not None and event.end_ms is not None:
                in_window = np.where((timestamps >= event.start_ms) & (timestamps <= event.end_ms))[0]
                target_idxs = in_window.tolist()

            for t_idx in target_idxs:
                if 0 <= t_idx < num_frames:
                    # Unary peak
                    scores[e_idx, t_idx] = 0.92
                    # Align frame visual vector with event text embedding + motion direction
                    embs[t_idx] = q_embs[e_idx] + rng.normal(0.0, 0.05, size=dim).astype(np.float32)

        # Normalize visual embeddings
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        video_scores[vid] = VideoEventScores(
            video_id=vid,
            frame_ids=frame_ids,
            frame_idx=frame_idx,
            timestamps_ms=timestamps,
            scores=scores,
        )
        frame_embeddings[vid] = embs

    # 2. Build distractor videos (unrelated content)
    # Use max events across queries
    max_events = max(len(q.events) for q in dataset.queries) if dataset.queries else 2
    for d in range(num_distractors):
        dist_id = f"V_DISTRACT_{d+1:02d}"
        dist_scores = rng.uniform(0.02, 0.15, size=(max_events, num_frames)).astype(np.float32)
        dist_ts = np.arange(num_frames, dtype=np.int64) * 100
        dist_idx = np.arange(num_frames, dtype=np.int64)
        dist_ids = np.array([f"{dist_id}_f{i:03d}" for i in range(num_frames)])

        dist_embs = rng.normal(size=(num_frames, dim)).astype(np.float32)
        norms = np.linalg.norm(dist_embs, axis=-1, keepdims=True)
        dist_embs = dist_embs / np.maximum(norms, 1e-12)

        video_scores[dist_id] = VideoEventScores(
            video_id=dist_id,
            frame_ids=dist_ids,
            frame_idx=dist_idx,
            timestamps_ms=dist_ts,
            scores=dist_scores,
        )
        frame_embeddings[dist_id] = dist_embs

    return video_scores, frame_embeddings
