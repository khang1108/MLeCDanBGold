"""Route and fuse full-corpus Dense and BM25 temporal evidence.

Original Vietnamese caption events are resolved by orchestration before this
scorer. This module never calls query preparation, shortlists frames, or
changes DP logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from hcmai.common.config import HybridTemporalConfig
from hcmai.retrieval.evidence.normalization import minmax_rows
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class TemporalEvidenceScorer:
    """Produce canonical per-video matrices for selected temporal evidence."""

    def __init__(
        self,
        *,
        visual_index: Any,
        dense: Any | None,
        bm25: Any | None,
        config: HybridTemporalConfig,
        visual_dense_ready: bool | None = None,
        context_dense_ready: bool | None = None,
        asr_dense_ready: bool | None = None,
    ) -> None:
        """Bind independently optional Dense and BM25 scoring capabilities."""

        self.visual_index = visual_index
        self.dense = dense
        self.bm25 = bm25
        self.config = config
        dense_ready = dense is not None
        self.visual_dense_ready = dense_ready if visual_dense_ready is None else visual_dense_ready
        self.context_dense_ready = (
            dense_ready if context_dense_ready is None else context_dense_ready
        )
        self.asr_dense_ready = dense_ready if asr_dense_ready is None else asr_dense_ready

    def score_events(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        caption_events: Sequence[str] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> list[VideoEventScores]:
        """Score enabled sources over every frame and split by canonical video."""

        if not use_dense and not use_bm25:
            raise ValueError("at least one temporal evidence source must be enabled")
        event_count = len(original_events)
        if not event_count or len(retrieval_events) != event_count:
            raise ValueError("original and retrieval event counts must match")
        if use_bm25 and (caption_events is None or len(caption_events) != event_count):
            raise ValueError("BM25 caption event counts must match original events")

        dense_scores: np.ndarray | None = None
        bm25_scores: np.ndarray | None = None
        if use_dense:
            if self.dense is None:
                raise RuntimeError("Dense temporal evidence is unavailable")
            dense_scores = np.asarray(self.dense.score_events(retrieval_events), dtype=np.float32)
        if use_bm25:
            if self.bm25 is None:
                raise RuntimeError("BM25 temporal evidence is unavailable")
            assert caption_events is not None
            bm25_scores = minmax_rows(self.bm25.score_events(original_events, caption_events))

        if dense_scores is not None and bm25_scores is not None:
            scores = self.config.dense_weight * dense_scores + self.config.bm25_weight * bm25_scores
        else:
            scores = dense_scores if dense_scores is not None else bm25_scores
        assert scores is not None
        expected_shape = (event_count, len(self.visual_index.frame_ids))
        if scores.shape != expected_shape:
            raise ValueError("temporal evidence matrix shape conflicts with canonical index")
        return _split_videos(self.visual_index, np.asarray(scores, dtype=np.float32))


def _split_videos(index: Any, scores: np.ndarray) -> list[VideoEventScores]:
    """Split a canonical full-corpus matrix using visual-index frame order."""

    video_ids = sorted({str(video_id) for video_id in index.video_ids})
    return [
        VideoEventScores(
            video_id=video_id,
            frame_ids=index.frame_ids[positions],
            frame_idx=index.frame_idx[positions],
            timestamps_ms=index.timestamps[positions],
            scores=scores[:, positions],
        )
        for video_id in video_ids
        if len(positions := index.video_positions(video_id))
    ]
