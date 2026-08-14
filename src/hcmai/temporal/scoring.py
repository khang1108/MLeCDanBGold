"""Score assembled scenes from their evidence, with every constant exposed as a knob."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from hcmai.temporal.models import (
    EvidencePoint,
    EvidenceStatus,
    RelationType,
    SceneCandidate,
    TemporalConstraint,
)
from hcmai.temporal.plan import TemporalQueryPlan


@dataclass(frozen=True, slots=True)
class SceneScorer:
    """Combine evidence quality, hint coverage, compactness and explicit relations."""

    semantic_weight: float = 0.4
    coverage_weight: float = 0.3
    temporal_weight: float = 0.15
    relation_weight: float = 0.15
    min_score_weight: float = 0.5
    compact_half_life_ms: int = 5_000
    discriminative_hint_weights: bool = False

    def __post_init__(self) -> None:
        weights = (
            self.semantic_weight,
            self.coverage_weight,
            self.temporal_weight,
            self.relation_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("scene score weights must be non-negative")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("scene score weights must sum to 1")
        if not 0.0 <= self.min_score_weight <= 1.0:
            raise ValueError("min_score_weight must be within [0, 1]")
        if self.compact_half_life_ms < 1:
            raise ValueError("compact_half_life_ms must be at least 1")

    def rank(
        self,
        plan: TemporalQueryPlan,
        scenes: Sequence[SceneCandidate],
    ) -> tuple[SceneCandidate, ...]:
        """Score one round's scenes against each other, strongest first."""
        scored = [self.score(plan, scene) for scene in scenes]
        # Semantic scores are RRF sums (~0.01-0.07); the peak brings them onto the [0, 1]
        # scale the other components already use, keeping their ratios unlike min-max.
        peak = max((scene.semantic_score for scene in scored), default=0.0)
        if peak > 0.0:
            scored = [
                self._finalized(scene, scene.semantic_score / peak) for scene in scored
            ]
        return tuple(
            sorted(
                scored,
                key=lambda scene: (
                    -scene.final_score,
                    scene.start_ms,
                    scene.end_ms,
                    scene.video_id,
                ),
            )
        )

    def score(self, plan: TemporalQueryPlan, scene: SceneCandidate) -> SceneCandidate:
        """Fill in every score component; ``final_score`` only ranks within one `rank` call."""
        matched: dict[str, tuple[EvidencePoint, ...]] = {}
        for unit in plan.units:
            evidence = scene.evidence_by_unit.get(unit.unit_id)
            if evidence is not None and evidence.status is EvidenceStatus.MATCHED and evidence.points:
                matched[unit.unit_id] = evidence.points
        if not matched:
            return replace(
                scene,
                unit_scores={},
                coverage_score=0.0,
                semantic_score=0.0,
                temporal_score=0.0,
                relation_score=0.0,
                final_score=0.0,
            )

        unit_scores = {
            unit_id: max(point.relevance_score for point in points)
            for unit_id, points in matched.items()
        }
        # A unit's own spread (top1 - mean) tilts its weight; otherwise every unit weighs 1.
        unit_weights: dict[str, float] = {}
        for unit_id, top in unit_scores.items():
            points = matched[unit_id]
            mean = sum(point.relevance_score for point in points) / len(points)
            unit_weights[unit_id] = 1.0 + top - mean if self.discriminative_hint_weights else 1.0
        weighted_mean = sum(
            unit_scores[unit_id] * weight for unit_id, weight in unit_weights.items()
        ) / sum(unit_weights.values())
        semantic = (1.0 - self.min_score_weight) * weighted_mean + self.min_score_weight * min(
            unit_scores.values()
        )
        coverage = len(unit_scores) / len(plan.units)
        # Soft decay on the span plus its widest internal gap, never a hard span cutoff.
        timestamps = sorted(
            point.timestamp_ms
            for evidence in scene.evidence_by_unit.values()
            for point in evidence.points
        )
        widest_gap = max(
            (later - earlier for earlier, later in zip(timestamps, timestamps[1:])),
            default=0,
        )
        temporal = self.compact_half_life_ms / (
            self.compact_half_life_ms + scene.end_ms - scene.start_ms + widest_gap
        )
        return self._finalized(
            replace(
                scene,
                unit_scores=unit_scores,
                coverage_score=coverage,
                temporal_score=temporal,
                relation_score=self._relation_score(plan.constraints, matched),
            ),
            semantic,
        )

    def _finalized(self, scene: SceneCandidate, semantic: float) -> SceneCandidate:
        """Attach the semantic score and the weighted sum it feeds."""
        return replace(
            scene,
            semantic_score=semantic,
            final_score=min(
                1.0,
                self.semantic_weight * semantic
                + self.coverage_weight * scene.coverage_score
                + self.temporal_weight * scene.temporal_score
                + self.relation_weight * scene.relation_score,
            ),
        )

    def _relation_score(
        self,
        constraints: Sequence[TemporalConstraint],
        matched: Mapping[str, tuple[EvidencePoint, ...]],
    ) -> float:
        """Share of explicit constraints that hold between the units' best frames."""
        best_timestamp = {
            unit_id: min(
                points,
                key=lambda point: (-point.relevance_score, point.frame_idx, point.frame_id),
            ).timestamp_ms
            for unit_id, points in matched.items()
        }
        applicable = [
            constraint
            for constraint in constraints
            if constraint.left_unit_id in best_timestamp
            and constraint.right_unit_id in best_timestamp
        ]
        if not applicable:
            return 1.0

        satisfied = 0
        for constraint in applicable:
            left = best_timestamp[constraint.left_unit_id]
            right = best_timestamp[constraint.right_unit_id]
            if constraint.relation is RelationType.BEFORE and left >= right:
                continue
            if constraint.relation is RelationType.AFTER and left <= right:
                continue
            # during / overlaps / same_scene: a hint has no duration, so both units being
            # inside the scene is everything this evidence can prove.
            satisfied += 1
        return satisfied / len(applicable)
