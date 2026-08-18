"""Transparent grounded joint ranking baseline."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from ..domain.models import GroundedAnswerCandidate


DEFAULT_WEIGHTS = {
    "video": 0.2, "frame": 0.2, "grounding": 0.2,
    "answer": 0.2, "consistency": 0.2,
}


def rank_grounded_answers(
    candidates: list[GroundedAnswerCandidate],
    *,
    weights: dict[str, float] | None = None,
) -> list[GroundedAnswerCandidate]:
    active = weights or DEFAULT_WEIGHTS
    if set(active) != set(DEFAULT_WEIGHTS) or any(value < 0 for value in active.values()):
        raise ValueError("weights must define non-negative video/frame/grounding/answer/consistency")
    valid = [item for item in candidates if item.grounded and item.answer and item.normalized_answer]
    if not valid:
        return []
    counts = Counter(item.normalized_answer for item in valid)
    fields = {
        "video": [item.video_score for item in valid],
        "frame": [item.frame_score for item in valid],
        "grounding": [0.5 * item.localization_score + 0.5 * item.evidence_coverage_score for item in valid],
        "answer": [item.answer_confidence for item in valid],
        "consistency": [counts[item.normalized_answer] / len(valid) for item in valid],
    }
    normalized = {name: _minmax(values) for name, values in fields.items()}
    ranked: list[GroundedAnswerCandidate] = []
    for index, item in enumerate(valid):
        components = {name: values[index] for name, values in normalized.items()}
        score = sum(active[name] * components[name] for name in active)
        ranked.append(replace(
            item, consistency_score=fields["consistency"][index],
            joint_score=score, score_components=components,
        ))
    return sorted(ranked, key=lambda item: (
        -item.joint_score, -item.answer_confidence, item.scene.video_id,
        item.scene.start_ms, item.evidence_frame_id, item.normalized_answer,
    ))


def _minmax(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]
