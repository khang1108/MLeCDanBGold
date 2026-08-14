"""Assemble Top-M evidence points into ranked scene windows for KIS/VQA."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hcmai.temporal.alignment.base import AlignmentResult
from hcmai.temporal.models import EvidencePoint, EvidenceSet, EvidenceStatus, SceneCandidate
from hcmai.temporal.plan import TemporalQueryPlan

_RankedScene = tuple[tuple[float, float, int, int, str], SceneCandidate]


@dataclass(frozen=True, slots=True)
class CoverageWindowAligner:
    """Rank compact windows that cover as many query units as the evidence allows."""

    max_span_ms: int = 30_000
    max_per_video: int = 5
    merge_overlap: float = 0.8

    def __post_init__(self) -> None:
        if self.max_span_ms < 0:
            raise ValueError("max_span_ms must be non-negative")
        if self.max_per_video < 1:
            raise ValueError("max_per_video must be at least 1")
        if not 0.0 < self.merge_overlap <= 1.0:
            raise ValueError("merge_overlap must be within (0, 1]")

    def align(
        self,
        plan: TemporalQueryPlan,
        evidence: Sequence[EvidencePoint],
    ) -> AlignmentResult[SceneCandidate]:
        if not plan.units:
            return AlignmentResult()

        unit_ids = {unit.unit_id for unit in plan.units}
        best_by_video: dict[str, dict[tuple[str, str, str, int], EvidencePoint]] = {}
        for point in evidence:
            if point.unit_id not in unit_ids:
                raise ValueError(f"evidence unit {point.unit_id} is not in the plan")
            best_by_key = best_by_video.setdefault(point.video_id, {})
            key = (point.unit_id, *point.canonical_identity)
            kept = best_by_key.get(key)
            if kept is None or point.relevance_score > kept.relevance_score:
                best_by_key[key] = point

        scenes = [
            scene
            for video_id, best_by_key in best_by_video.items()
            for scene in self._video_scenes(video_id, best_by_key.values(), len(plan.units))
        ]
        return AlignmentResult(
            candidates=tuple(scene for _, scene in sorted(scenes, key=lambda ranked: ranked[0]))
        )

    def _video_scenes(
        self,
        video_id: str,
        points: Iterable[EvidencePoint],
        total_units: int,
    ) -> list[_RankedScene]:
        ordered = sorted(
            points,
            key=lambda point: (
                point.timestamp_ms,
                point.frame_idx,
                point.frame_id,
                point.unit_id,
            ),
        )
        timestamps = [point.timestamp_ms for point in ordered]
        # One shrink walk per start: quadratic, but a video holds at most units x top_m points.
        windows: list[tuple[EvidencePoint, ...]] = []
        for start, timestamp in enumerate(timestamps):
            reach = bisect_right(timestamps, timestamp + self.max_span_ms)
            reachable = {point.unit_id for point in ordered[start:reach]}
            end = start
            covered = {ordered[start].unit_id}
            while covered != reachable:
                end += 1
                covered.add(ordered[end].unit_id)
            windows.append(tuple(ordered[start : end + 1]))

        kept: list[_RankedScene] = []
        for ranked in sorted(
            (self._scene(video_id, window, total_units) for window in windows),
            key=lambda ranked: ranked[0],
        ):
            if any(self._nearly_equal(ranked[1], scene) for _, scene in kept):
                continue
            kept.append(ranked)
            if len(kept) == self.max_per_video:
                break
        return kept

    def _scene(
        self,
        video_id: str,
        window: tuple[EvidencePoint, ...],
        total_units: int,
    ) -> _RankedScene:
        points_by_unit: dict[str, list[EvidencePoint]] = {}
        for point in window:
            points_by_unit.setdefault(point.unit_id, []).append(point)
        scene = SceneCandidate(
            video_id=video_id,
            start_ms=window[0].timestamp_ms,
            end_ms=window[-1].timestamp_ms,
            evidence_by_unit={
                unit_id: EvidenceSet(
                    status=EvidenceStatus.MATCHED,
                    points=tuple(
                        sorted(
                            unit_points,
                            key=lambda point: (
                                -point.relevance_score,
                                point.frame_idx,
                                point.frame_id,
                            ),
                        )
                    ),
                )
                for unit_id, unit_points in points_by_unit.items()
            },
            coverage_score=len(points_by_unit) / total_units,
        )
        strength = sum(
            max(point.relevance_score for point in unit_points)
            for unit_points in points_by_unit.values()
        )
        return (
            (
                -scene.coverage_score,
                -strength,
                scene.end_ms - scene.start_ms,
                scene.start_ms,
                video_id,
            ),
            scene,
        )

    def _nearly_equal(self, scene: SceneCandidate, other: SceneCandidate) -> bool:
        """True when both windows describe the same moment, so only the stronger is kept."""
        overlap = min(scene.end_ms, other.end_ms) - max(scene.start_ms, other.start_ms)
        if overlap < 0:
            return False
        shortest = min(scene.end_ms - scene.start_ms, other.end_ms - other.start_ms)
        return shortest == 0 or overlap / shortest >= self.merge_overlap
