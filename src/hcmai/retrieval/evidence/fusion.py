"""Event-adaptive, coverage-aware multimodal temporal evidence fusion.

This module combines independent evidence components (Dense visual/context/ASR,
BM25 title/caption/OCR/ASR) into a single unified temporal score matrix
[event_count, frame_count]. Fusion dynamically routes weights based on event
text cues, scales them by calibrated component reliability, and renormalizes
locally over modalities actually available and covering each frame.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from hcmai.common.config import AdaptiveTemporalFusionConfig
from hcmai.retrieval.evidence.calibration import calibrate_component
from hcmai.retrieval.evidence.components import TemporalScoreBundle

SPEECH_CUES: tuple[str, ...] = (
    "nói", "hỏi", "trả lời", "đối thoại", "phỏng vấn", "cho biết", "giới thiệu",
    "says", "asks", "answers", "talks", "speaks", "interview", "announces",
)
OCR_CUES: tuple[str, ...] = (
    "dòng chữ", "chữ", "biển hiệu", "màn hình hiển thị", "logo", "nhãn",
    "text", "sign", "screen displays", "logo", "label",
)
VISUAL_CUES: tuple[str, ...] = (
    "mặc", "cầm", "đặt", "đứng", "ngồi", "chạy", "xe", "đĩa", "màu",
    "wearing", "holds", "places", "stands", "sits", "runs", "plate", "color",
)


class EventModalityRouter:
    """Deterministic routing of component weight multipliers based on text cues."""

    def __init__(self, config: AdaptiveTemporalFusionConfig) -> None:
        self.config = config

    def multipliers(
        self,
        original_event: str,
        retrieval_event: str,
    ) -> dict[str, float]:
        """Compute unnormalized positive component weights for one event query.

        Args:
            original_event: Original natural-language query string (e.g. Vietnamese).
            retrieval_event: Translated / prepared retrieval string (e.g. English).

        Returns:
            Dictionary mapping component names to unnormalized positive weights.
        """

        weights = dict(self.config.base_component_weights)
        if not self.config.event_routing:
            return weights

        combined = f"{original_event.lower()} {retrieval_event.lower()}"

        if any(cue in combined for cue in SPEECH_CUES):
            if "asr_dense" in weights:
                weights["asr_dense"] *= self.config.speech_boost
            if "bm25_asr" in weights:
                weights["bm25_asr"] *= self.config.speech_boost

        if any(cue in combined for cue in OCR_CUES):
            if "bm25_ocr" in weights:
                weights["bm25_ocr"] *= self.config.ocr_boost

        if any(cue in combined for cue in VISUAL_CUES):
            if "visual_dense" in weights:
                weights["visual_dense"] *= self.config.visual_boost
            if "context_dense" in weights:
                weights["context_dense"] *= self.config.visual_boost

        return weights


class TemporalFusionScorer:
    """Adaptive multimodal evidence fusion engine for temporal scoring."""

    def __init__(self, config: AdaptiveTemporalFusionConfig) -> None:
        self.config = config
        self.router = EventModalityRouter(config)

    def fuse(
        self,
        *,
        original_events: Sequence[str],
        retrieval_events: Sequence[str],
        bundle: TemporalScoreBundle,
    ) -> np.ndarray:
        """Fuse multimodal temporal score components into a unified score matrix.

        Args:
            original_events: Original event queries.
            retrieval_events: Retrieval event queries matching original_events.
            bundle: Bundle of raw score components and coverage masks.

        Returns:
            Float32 score matrix shaped ``[len(original_events), frame_count]``.

        Raises:
            ValueError: If event counts do not match bundle dimensions.
        """

        if len(original_events) != len(retrieval_events):
            raise ValueError("original and retrieval event counts must match")
        if bundle.shape[0] != len(original_events):
            raise ValueError("component event count must match query event count")

        calibrated = {
            name: calibrate_component(component.raw_scores, self.config.calibration)
            for name, component in bundle.components.items()
        }
        result = np.zeros(bundle.shape, dtype=np.float32)

        for event_index, (original, retrieval) in enumerate(
            zip(original_events, retrieval_events, strict=True)
        ):
            requested = self.router.multipliers(original, retrieval)
            numerator = np.zeros(bundle.shape[1], dtype=np.float32)
            denominator = np.zeros(bundle.shape[1], dtype=np.float32)

            for name, component in bundle.components.items():
                base = float(requested.get(name, 0.0))
                if base <= 0.0:
                    continue
                confidence = (
                    float(calibrated[name].reliability[event_index])
                    if self.config.confidence_gating
                    else 1.0
                )
                weight = base * confidence
                if weight <= 0.0:
                    continue
                coverage = (
                    np.ones(bundle.shape[1], dtype=np.float32)
                    if component.coverage is None
                    else component.coverage.astype(np.float32)
                )
                effective = weight * coverage
                numerator += effective * calibrated[name].scores[event_index]
                denominator += effective

            np.divide(
                numerator,
                denominator,
                out=result[event_index],
                where=denominator > 0.0,
            )
            missing = denominator <= 0.0
            if np.any(missing) and "visual_dense" in calibrated:
                result[event_index, missing] = calibrated["visual_dense"].scores[
                    event_index, missing
                ]

        return result


__all__ = [
    "EventModalityRouter",
    "OCR_CUES",
    "SPEECH_CUES",
    "TemporalFusionScorer",
    "VISUAL_CUES",
]
