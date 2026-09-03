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
from hcmai.retrieval.evidence.calibration import CalibratedComponent
from hcmai.retrieval.evidence.components import (
    TemporalScoreBundle,
    TemporalScoreComponent,
)
from hcmai.retrieval.evidence.diagnostics import (
    TemporalEvidenceDebugResult,
    build_evidence_diagnostics,
)
from hcmai.retrieval.evidence.fusion import TemporalFusionScorer
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

    def _adaptive_scorer(self) -> TemporalFusionScorer:
        """Instantiate a stateless adaptive scorer using current configuration."""

        return TemporalFusionScorer(self.config.adaptive)

    def with_config(self, config: HybridTemporalConfig) -> TemporalEvidenceScorer:
        """Create an immutable clone of this evidence scorer with updated configuration."""

        return TemporalEvidenceScorer(
            visual_index=self.visual_index,
            dense=self.dense,
            bm25=self.bm25,
            config=config,
            visual_dense_ready=self.visual_dense_ready,
            context_dense_ready=self.context_dense_ready,
            asr_dense_ready=self.asr_dense_ready,
        )

    def _score_legacy(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        caption_events: Sequence[str] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> np.ndarray:
        """Score events using legacy fixed-weight normalized fusion."""

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
        return np.asarray(scores, dtype=np.float32)

    def _score_components(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        caption_events: Sequence[str] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> TemporalScoreBundle:
        """Collect raw score components from enabled evidence scorers."""

        components: dict[str, TemporalScoreComponent] = {}
        if use_dense:
            if self.dense is None:
                raise RuntimeError("Dense temporal evidence is unavailable")
            interval_proj = getattr(self.config.adaptive, "asr_interval_projection", True)
            try:
                dense_bundle = self.dense.score_components(
                    retrieval_events, asr_interval_projection=interval_proj
                )
            except TypeError:
                dense_bundle = self.dense.score_components(retrieval_events)
            components.update(dense_bundle.components)
        if use_bm25:
            if self.bm25 is None:
                raise RuntimeError("BM25 temporal evidence is unavailable")
            assert caption_events is not None
            components.update(
                self.bm25.score_components(original_events, caption_events).components
            )
        return TemporalScoreBundle(components)

    def _prepare_adaptive_components(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        caption_events: Sequence[str] | None,
        use_dense: bool,
        use_bm25: bool,
    ) -> tuple[TemporalScoreBundle, dict[str, CalibratedComponent], np.ndarray]:
        """Score and calibrate components and compute fused scores."""

        bundle = self._score_components(
            original_events,
            retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        scorer = self._adaptive_scorer()
        calibrated = scorer.calibrate_bundle(bundle)
        fused = scorer.fuse(
            original_events=original_events,
            retrieval_events=retrieval_events,
            bundle=bundle,
        )
        return bundle, calibrated, fused

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

        if self.config.fusion_mode == "legacy":
            scores = self._score_legacy(
                original_events,
                retrieval_events,
                caption_events=caption_events,
                use_dense=use_dense,
                use_bm25=use_bm25,
            )
        else:
            _, _, scores = self._prepare_adaptive_components(
                original_events,
                retrieval_events,
                caption_events=caption_events,
                use_dense=use_dense,
                use_bm25=use_bm25,
            )

        expected_shape = (event_count, len(self.visual_index.frame_ids))
        if scores.shape != expected_shape:
            raise ValueError("temporal evidence matrix shape conflicts with canonical index")
        return _split_videos(self.visual_index, np.asarray(scores, dtype=np.float32))

    def debug_score_events(
        self,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        *,
        caption_events: Sequence[str] | None = None,
        use_dense: bool = True,
        use_bm25: bool = True,
        top_positions: int = 10,
    ) -> TemporalEvidenceDebugResult:
        """Score full corpus and return component-level diagnostic telemetry."""

        if not use_dense and not use_bm25:
            raise ValueError("at least one temporal evidence source must be enabled")
        event_count = len(original_events)
        if not event_count or len(retrieval_events) != event_count:
            raise ValueError("original and retrieval event counts must match")
        if use_bm25 and (caption_events is None or len(caption_events) != event_count):
            raise ValueError("BM25 caption event counts must match original events")

        bundle, calibrated, fused = self._prepare_adaptive_components(
            original_events,
            retrieval_events,
            caption_events=caption_events,
            use_dense=use_dense,
            use_bm25=use_bm25,
        )
        return build_evidence_diagnostics(
            bundle=bundle,
            calibrated=calibrated,
            fused_scores=fused,
            top_positions=top_positions,
        )


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
