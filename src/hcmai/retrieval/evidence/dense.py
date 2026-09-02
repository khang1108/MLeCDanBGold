"""Full-corpus multimodal Dense temporal evidence scoring.

The scorer batches one SigLIP text encoding and one shared BGE encoding per
request, then scores Visual, Context, and frame-ASR indexes without shortlists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from hcmai.common.config import DenseTemporalWeights
from hcmai.retrieval.evidence.normalization import minmax_rows


class DenseTemporalScorer:
    """Combine normalized Visual, Context, and frame-ASR Dense score rows."""

    def __init__(
        self,
        *,
        visual_index: Any,
        context_index: Any,
        asr_index: Any,
        visual_encoder: Any,
        text_encoder: Any,
        weights: DenseTemporalWeights,
        chunk_size: int = 65_536,
    ) -> None:
        """Bind three canonical indexes and two non-duplicated query encoders."""

        _validate_indexes(visual_index, context_index, asr_index)
        self.visual_index = visual_index
        self.context_index = context_index
        self.asr_index = asr_index
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.weights = weights
        self.chunk_size = chunk_size

    def score_events(self, retrieval_events: Sequence[str]) -> np.ndarray:
        """Score every canonical frame for all selected retrieval events."""

        events = [" ".join(event.split()) for event in retrieval_events]
        if not events or any(not event for event in events):
            raise ValueError("retrieval events must contain non-empty strings")
        visual_vectors = np.asarray(self.visual_encoder.encode_text(events), dtype=np.float32)
        text_vectors = np.asarray(self.text_encoder.encode_text(events), dtype=np.float32)
        positions = np.arange(len(self.visual_index.frame_ids), dtype=np.int64)
        visual = minmax_rows(
            self.visual_index.score_subset(visual_vectors, positions, self.chunk_size)
        )
        context = minmax_rows(
            self.context_index.score_subset(text_vectors, positions, self.chunk_size)
        )
        asr = minmax_rows(self.asr_index.score_subset(text_vectors, positions, self.chunk_size))
        return np.asarray(
            self.weights.visual_weight * visual
            + self.weights.context_weight * context
            + self.weights.asr_weight * asr,
            dtype=np.float32,
        )


def _validate_indexes(visual: Any, context: Any, asr: Any) -> None:
    """Require identical canonical identity order and compatible dimensions."""

    expected = _identity(visual)
    for name, index in (("context", context), ("asr", asr)):
        if _identity(index) != expected:
            raise ValueError(f"{name} Dense index identity conflicts with visual index")
    if context.metadata.embedding_dim != asr.metadata.embedding_dim:
        raise ValueError("Context and ASR Dense index dimensions differ")


def _identity(index: Any) -> tuple[tuple[str, str, int, int], ...]:
    """Materialize the canonical identity tuple for one Dense index."""

    lengths = {
        len(index.frame_ids),
        len(index.video_ids),
        len(index.frame_idx),
        len(index.timestamps),
    }
    if len(lengths) != 1:
        raise ValueError("Dense index identity arrays have unequal lengths")
    return tuple(
        (str(frame_id), str(video_id), int(frame_idx), int(timestamp_ms))
        for frame_id, video_id, frame_idx, timestamp_ms in zip(
            index.frame_ids,
            index.video_ids,
            index.frame_idx,
            index.timestamps,
            strict=True,
    ))