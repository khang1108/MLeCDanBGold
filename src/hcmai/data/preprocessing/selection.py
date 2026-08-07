"""Adaptive candidate selection and optional DINOv3 deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.video import FrameMeta, add_dynamic_coverage, peak_indices


@dataclass(slots=True)
class CandidateFrame:
    """One selected frame and the signals that selected it."""
    frame: FrameMeta
    shot_id: int
    shot_score: float
    event_score: float
    reasons: tuple[str, ...]
    protected: bool


class DinoEncoder:
    """Lazy DINOv3 image encoder used only for local deduplication."""
    def __init__(self, config: PreprocessingConfig) -> None:
        """Keep model settings until the first candidate batch."""
        self.config = config
        self.processor, self.model = None, None

    def encode(self, images: list[Any]) -> np.ndarray:
        """Return normalized global DINOv3 embeddings."""
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.config.dino_model)
            dtype = getattr(torch, self.config.dino_dtype)
            self.model = AutoModel.from_pretrained(self.config.dino_model,
                torch_dtype=dtype).to(self.config.dino_device).eval()
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.config.dino_device
        )
        with torch.inference_mode():
            output = self.model(**inputs)
        vectors = torch.nn.functional.normalize(output.pooler_output.float(), dim=1)
        return vectors.cpu().numpy()


def _expand_burst(frames: list[FrameMeta], center: int, reason: str,
                  config: PreprocessingConfig, reasons: list[set[str]]) -> None:
    """Add regularly spaced context around one strong trigger."""
    center_ms = frames[center].timestamp_ms
    last_ms = -config.burst_step_ms
    for index, frame in enumerate(frames):
        if abs(frame.timestamp_ms - center_ms) > config.burst_radius_ms:
            continue
        if index == center or frame.timestamp_ms - last_ms >= config.burst_step_ms:
            reasons[index].add(f"{reason}_context")
            last_ms = frame.timestamp_ms


def select_candidates(frames: list[FrameMeta], shot_scores: np.ndarray,
                      event_scores: np.ndarray,
                      config: PreprocessingConfig) -> list[CandidateFrame]:
    """Union independent triggers and enforce dynamic temporal coverage."""
    if not frames:
        return []
    if len(shot_scores) != len(frames) or len(event_scores) != len(frames):
        raise ValueError("Boundary score count does not match decoded frames")
    reasons = [set() for _ in frames]
    protected: set[int] = {0, len(frames) - 1}
    shot_peaks = peak_indices(shot_scores, config.shot_threshold)
    event_peaks = peak_indices(event_scores, config.event_threshold)
    reasons[0].add("coverage_anchor")
    reasons[-1].add("coverage_anchor")
    for index, frame in enumerate(frames):
        triggers: list[str] = []
        if index in shot_peaks:
            triggers.append("shot_boundary")
        if index in event_peaks:
            triggers.append("event_boundary")
        neighbors = frames[max(0, index - 1): index + 2]
        if frame.motion_score >= config.motion_threshold and frame.motion_score == max(
            item.motion_score for item in neighbors
        ):
            triggers.append("motion_peak")
        for trigger in triggers:
            reasons[index].add(trigger)
            protected.add(index)
            _expand_burst(frames, index, trigger.removesuffix("_boundary"), config, reasons)
    add_dynamic_coverage(frames, reasons, protected, config)
    shot_id = 0
    selected: list[CandidateFrame] = []
    for index, (frame, frame_reasons) in enumerate(zip(frames, reasons)):
        if index in shot_peaks and index > 0:
            shot_id += 1
        if frame_reasons:
            selected.append(CandidateFrame(
                frame, shot_id, float(shot_scores[index]), float(event_scores[index]),
                tuple(sorted(frame_reasons)), index in protected))
    return selected


def deduplicate(candidates: list[CandidateFrame], embeddings: np.ndarray,
                config: PreprocessingConfig) -> list[CandidateFrame]:
    """Drop only nearby, same-shot, unprotected semantic duplicates."""
    kept: list[int] = []
    for index, candidate in enumerate(candidates):
        if not kept or candidate.protected:
            kept.append(index)
            continue
        previous = candidates[kept[-1]]
        similarity = float(embeddings[index] @ embeddings[kept[-1]])
        duplicate = (
            candidate.shot_id == previous.shot_id
            and candidate.frame.timestamp_ms - previous.frame.timestamp_ms
            <= config.dedup_window_ms
            and candidate.frame.motion_score <= config.dedup_motion_threshold
            and similarity >= config.dedup_similarity
        )
        if not duplicate:
            kept.append(index)
    return [candidates[index] for index in kept]
