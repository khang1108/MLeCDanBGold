"""Deterministic local metrics for grounded competition VQA."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

from hcmai.common.schemas import VQASubmission

VQA_CUTOFFS = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class VQAGold:
    """One frozen local annotation without assuming organizer normalization."""

    video_id: str
    start_frame_idx: int
    end_frame_idx: int
    normalized_answers: frozenset[str]

    def __post_init__(self) -> None:
        if self.start_frame_idx < 0 or self.end_frame_idx < self.start_frame_idx:
            raise ValueError("invalid accepted frame interval")
        if not self.normalized_answers:
            raise ValueError("at least one normalized answer is required")


@dataclass(frozen=True)
class VQAMetrics:
    """Per-query correctness and AIC-style Top-k summary."""

    correct_video: float
    frame_interval: float
    normalized_answer: float
    joint: float
    top_k_joint: Mapping[int, float]
    mean_top_k_score: float


def evaluate_vqa(
    submissions: Sequence[VQASubmission],
    gold: VQAGold,
    *,
    cutoffs: Iterable[int] = VQA_CUTOFFS,
) -> VQAMetrics:
    """Evaluate ranked grounded rows using explicit local answer aliases."""

    rows = list(submissions)
    video_hits = [row.video_id == gold.video_id for row in rows]
    frame_hits = [
        video_hit
        and gold.start_frame_idx <= row.frame_idx <= gold.end_frame_idx
        for row, video_hit in zip(rows, video_hits)
    ]
    answer_hits = [
        (row.normalized_answer or row.answer) in gold.normalized_answers
        for row in rows
    ]
    joint_hits = [
        video and frame and answer
        for video, frame, answer in zip(video_hits, frame_hits, answer_hits)
    ]
    ordered_cutoffs = tuple(cutoffs)
    if not ordered_cutoffs or any(value < 1 for value in ordered_cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    top_k = {
        cutoff: float(any(joint_hits[:cutoff]))
        for cutoff in ordered_cutoffs
    }
    return VQAMetrics(
        correct_video=float(any(video_hits)),
        frame_interval=float(any(frame_hits)),
        normalized_answer=float(any(answer_hits)),
        joint=float(any(joint_hits)),
        top_k_joint=top_k,
        mean_top_k_score=mean(top_k.values()),
    )
