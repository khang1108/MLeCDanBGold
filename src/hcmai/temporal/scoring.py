"""Stable, config-driven scene score components and deterministic ranking."""

from __future__ import annotations

from collections import defaultdict

from hcmai.common.config import ProgressiveSearchConfig
from hcmai.common.schemas import QueryUnit, SceneCandidate, TemporalConstraint

from .evidence import ProgressiveEvidenceState
from .relations import relation_satisfaction


def score_scene(
    scene: SceneCandidate,
    units: list[QueryUnit],
    state: ProgressiveEvidenceState,
    constraints: list[TemporalConstraint],
    config: ProgressiveSearchConfig,
    *,
    coherence_window_ms: int,
    unit_score_bounds: dict[str, tuple[float, float]] | None = None,
) -> SceneCandidate:
    """Compute named public components without converting UNKNOWN to zero."""

    scores_by_unit: dict[str, list[float]] = defaultdict(list)
    for item in scene.evidence:
        for unit_id, score in item.unit_scores.items():
            scores_by_unit[unit_id].append(
                _normalize_semantic_score(
                    score,
                    (unit_score_bounds or {}).get(unit_id),
                )
            )
    matched_scores = [max(values) for values in scores_by_unit.values() if values]
    semantic = (
        sum(matched_scores) / len(matched_scores)
        if matched_scores
        else 0.0
    )

    # Match quality and knowledge completeness are distinct. UNKNOWN is not a
    # no-match, but a one-of-four evaluated candidate is not treated as fully
    # known either.
    evaluable = [
        unit.unit_id
        for unit in units
        if state.is_evaluated(unit.unit_id, scene.video_id)
    ]
    matched = [unit_id for unit_id in evaluable if scores_by_unit.get(unit_id)]
    coverage = len(matched) / len(evaluable) if evaluable else 0.0
    evaluation_coverage = len(evaluable) / len(units) if units else 0.0
    effective_coverage = coverage * evaluation_coverage

    span = max(0, scene.end_ms - scene.start_ms)
    scale = max(1, coherence_window_ms)
    temporal = 1.0 / (1.0 + span / scale)
    relation, relation_labels = relation_satisfaction(
        constraints,
        scene.evidence,
        near_ms=scale,
    )
    active_components = [
        (config.scene_semantic_weight, semantic),
        (config.scene_coverage_weight, effective_coverage),
        (config.scene_temporal_weight, temporal),
    ]
    if relation is not None:
        active_components.append((config.scene_relation_weight, relation))
    active_components = [item for item in active_components if item[0] > 0]
    weighted_sum = sum(weight * value for weight, value in active_components)
    final = weighted_sum / sum(weight for weight, _ in active_components)
    labels = tuple(dict.fromkeys((
        *scene.reason_labels,
        f"matched_units:{len(matched)}",
        f"evaluable_units:{len(evaluable)}",
        f"total_units:{len(units)}",
        *relation_labels,
    )))
    unit_scores = {
        unit_id: max(values)
        for unit_id, values in scores_by_unit.items()
    }
    return scene.model_copy(
        update={
            "unit_scores": unit_scores,
            "semantic_score": _clamp(semantic),
            "coverage_score": _clamp(coverage),
            "evaluation_coverage_score": _clamp(evaluation_coverage),
            "temporal_score": _clamp(temporal),
            "relation_score": _clamp(relation) if relation is not None else None,
            "final_score": _clamp(final),
            "reason_labels": labels,
        }
    )


def rank_scenes(scenes: list[SceneCandidate]) -> list[SceneCandidate]:
    """Sort scenes using the frozen deterministic tie-break sequence."""

    return sorted(
        scenes,
        key=lambda scene: (
            -scene.final_score,
            -scene.semantic_score,
            -scene.coverage_score,
            -scene.evaluation_coverage_score,
            -scene.temporal_score,
            scene.video_id,
            scene.start_ms,
            scene.scene_id,
        ),
    )


def unit_score_bounds(
    state: ProgressiveEvidenceState,
    allowed_video_ids: set[str],
) -> dict[str, tuple[float, float]]:
    """Return per-query-unit ranges without assuming calibrated raw scores."""

    values: dict[str, list[float]] = {}
    for (unit_id, video_id), items in state.evidence.items():
        if video_id not in allowed_video_ids or not items:
            continue
        values.setdefault(unit_id, []).append(max(item.score for item in items))
    return {
        unit_id: (min(0.0, min(unit_values)), max(unit_values))
        for unit_id, unit_values in values.items()
    }


def normalize_score(value: float, bounds: tuple[float, float]) -> float:
    """Normalize one score inside its query-unit evidence range."""

    low, high = bounds
    if high <= low:
        return 1.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _clamp(value: float) -> float:
    """Normalize a public scene-score component to the frozen [0, 1] range."""

    return min(1.0, max(0.0, float(value)))


def _normalize_semantic_score(
    value: float,
    bounds: tuple[float, float] | None,
) -> float:
    """Map one retrieval score to a comparable per-unit [0, 1] scale."""

    if bounds is None:
        return _clamp(value)
    low, high = bounds
    if high <= low:
        return 1.0
    return _clamp((float(value) - low) / (high - low))
