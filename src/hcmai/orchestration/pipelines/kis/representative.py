"""Pick the one frame that represents a scene, using only that scene's own evidence."""

from __future__ import annotations

from dataclasses import dataclass

from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.temporal.models import EvidencePoint, SceneCandidate


@dataclass(frozen=True, slots=True)
class RepresentativeFrameSelector:
    """Score a scene's evidence frames, favouring units whose top frame stands out."""

    discriminative_weights: bool = True

    def select(self, scene: SceneCandidate) -> RetrievalCandidate | None:
        """Return the scene's representative frame, or None when it holds no evidence."""
        ranked = self.rank(scene)
        return ranked[0] if ranked else None

    def rank(self, scene: SceneCandidate) -> list[RetrievalCandidate]:
        """The scene's evidence frames, best representative first."""
        scored: dict[tuple[str, str, int], tuple[float, EvidencePoint]] = {}
        for evidence in scene.evidence_by_unit.values():
            # Spread between a unit's best frame and its own tail; 1.0 when it has no tail.
            scores = [point.relevance_score for point in evidence.points]
            weight = (
                1.0 + max(scores) - sum(scores) / len(scores)
                if self.discriminative_weights and scores
                else 1.0
            )
            for point in evidence.points:
                value, kept = scored.get(point.canonical_identity, (0.0, point))
                scored[point.canonical_identity] = (
                    value + weight * point.relevance_score,
                    kept if kept.relevance_score >= point.relevance_score else point,
                )
        # A frame serving several units accumulates their weights, so it wins on merit.
        return [
            RetrievalCandidate(
                frame_id=point.frame_id,
                source_scores={
                    RetrievalSource(name): score
                    for name, score in point.source_scores.items()
                },
                fusion_score=point.relevance_score,
                final_score=scene.final_score,
            )
            for _, point in sorted(
                scored.values(),
                key=lambda entry: (-entry[0], entry[1].frame_idx, entry[1].frame_id),
            )
        ]
