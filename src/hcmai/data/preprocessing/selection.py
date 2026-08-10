"""Select informative frames and remove local semantic duplicates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.video import (
    FrameMeta,
    add_dynamic_coverage,
    peak_indices,
)

BURST_RADIUS_MS = 500
BURST_STEP_MS = 200
DEDUP_WINDOW_MS = 1_000
DEDUP_MOTION_THRESHOLD = 0.008


@dataclass(slots=True)
class CandidateFrame:
    """One selected frame with its selection signals."""

    frame: FrameMeta
    shot_id: int
    shot_score: float
    event_score: float
    reasons: tuple[str, ...]
    protected: bool


class DinoEncoder:
    """Lazy DINO encoder for local semantic deduplication."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None

    def encode(self, images: list[Any]) -> np.ndarray:
        """Return normalized global image embeddings."""
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.dino_model
            )
            self.model = AutoModel.from_pretrained(
                self.config.dino_model,
                dtype=getattr(torch, self.config.dino_dtype),
            ).to(self.config.device).eval()
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.config.device
        )
        with torch.inference_mode():
            output = self.model(**inputs)
        vectors = torch.nn.functional.normalize(output.pooler_output.float(), dim=1)
        return vectors.cpu().numpy()


def _expand_burst(
    frames: list[FrameMeta], center: int, reason: str,
    reasons: list[set[str]],
) -> None:
    """Keep regularly spaced context around one trigger."""
    center_ms = frames[center].timestamp_ms
    last_ms = -BURST_STEP_MS
    for index, frame in enumerate(frames):
        if (
            abs(frame.timestamp_ms - center_ms) <= BURST_RADIUS_MS
            and (index == center or frame.timestamp_ms - last_ms >= BURST_STEP_MS)
        ):
            reasons[index].add(f"{reason}_context")
            last_ms = frame.timestamp_ms


def select_candidates(
    frames: list[FrameMeta],
    shot_scores: np.ndarray,
    event_scores: np.ndarray,
    config: PreprocessingConfig,
) -> list[CandidateFrame]:
    """Combine boundary, motion, context, and temporal coverage signals."""
    if not frames:
        return []
    if len(shot_scores) != len(frames) or len(event_scores) != len(frames):
        raise ValueError("Boundary score count does not match decoded frames")

    reasons = [set() for _ in frames]
    protected = {0, len(frames) - 1}
    shot_peaks = peak_indices(shot_scores, config.shot_threshold)
    event_peaks = peak_indices(event_scores, config.event_threshold)
    motion_peaks = {
        index
        for index, frame in enumerate(frames)
        if frame.motion_score >= config.motion_threshold
        and frame.motion_score == max(
            item.motion_score for item in frames[max(0, index - 1) : index + 2]
        )
    }
    reasons[0].add("coverage_anchor")
    reasons[-1].add("coverage_anchor")

    triggers = (
        ("shot_boundary", shot_peaks),
        ("event_boundary", event_peaks),
        ("motion_peak", motion_peaks),
    )
    for trigger, indices in triggers:
        for index in indices:
            reasons[index].add(trigger)
            protected.add(index)
            _expand_burst(
                frames, index, trigger.removesuffix("_boundary"), reasons
            )

    add_dynamic_coverage(frames, reasons, protected, config)
    selected = []
    shot_id = 0
    for index, (frame, frame_reasons) in enumerate(zip(frames, reasons)):
        if index in shot_peaks and index > 0:
            shot_id += 1
        if frame_reasons:
            selected.append(CandidateFrame(
                frame, shot_id, float(shot_scores[index]),
                float(event_scores[index]), tuple(sorted(frame_reasons)),
                index in protected,
            ))
    return selected


def deduplicate(
    candidates: list[CandidateFrame], embeddings: np.ndarray,
    config: PreprocessingConfig,
) -> list[CandidateFrame]:
    """Drop only nearby, same-shot, unprotected semantic duplicates."""
    kept: list[int] = []
    for index, candidate in enumerate(candidates):
        if not kept or candidate.protected:
            kept.append(index)
            continue
        previous = candidates[kept[-1]]
        duplicate = (
            candidate.shot_id == previous.shot_id
            and candidate.frame.timestamp_ms - previous.frame.timestamp_ms
            <= DEDUP_WINDOW_MS
            and candidate.frame.motion_score <= DEDUP_MOTION_THRESHOLD
            and float(embeddings[index] @ embeddings[kept[-1]])
            >= config.dedup_similarity
        )
        if not duplicate:
            kept.append(index)
    return [candidates[index] for index in kept]
