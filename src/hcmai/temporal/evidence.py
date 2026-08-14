from __future__ import annotations

from collections.abc import Iterable

from hcmai.temporal.models import EvidencePoint, EvidenceSet, EvidenceStatus


class EvidenceStore:
    """Top-M evidence points per `(unit_id, video_id)`, deduplicated by frame identity."""

    def __init__(self, top_m: int = 10) -> None:
        if top_m < 1:
            raise ValueError("top_m must be at least 1")
        self.top_m = top_m
        self._sets: dict[tuple[str, str], EvidenceSet] = {}

    def get(self, unit_id: str, video_id: str) -> EvidenceSet:
        """Return stored evidence, `UNKNOWN` when the pair was never evaluated."""
        return self._sets.get((unit_id, video_id), EvidenceSet())

    def record(
        self, unit_id: str, video_id: str, points: Iterable[EvidencePoint]
    ) -> EvidenceSet:
        """Merge retrieved points into the pair and mark it evaluated."""
        merged: dict[tuple[str, str, int], EvidencePoint] = {
            point.canonical_identity: point
            for point in self.get(unit_id, video_id).points
        }
        for point in points:
            if point.unit_id != unit_id or point.video_id != video_id:
                raise ValueError(
                    f"point ({point.unit_id}, {point.video_id}) does not belong to "
                    f"({unit_id}, {video_id})"
                )
            kept = merged.get(point.canonical_identity)
            if kept is None or point.relevance_score > kept.relevance_score:
                merged[point.canonical_identity] = point

        ranked = sorted(
            merged.values(),
            key=lambda point: (-point.relevance_score, point.frame_idx, point.frame_id),
        )[: self.top_m]
        status = EvidenceStatus.MATCHED if ranked else EvidenceStatus.EVALUATED_NO_MATCH
        evidence = EvidenceSet(status=status, points=tuple(ranked))
        self._sets[(unit_id, video_id)] = evidence
        return evidence

    def video_ids(self) -> tuple[str, ...]:
        """Return every video evaluated so far, in first-seen order."""
        return tuple(dict.fromkeys(video_id for _, video_id in self._sets))

    def unknown_units(self, unit_ids: Iterable[str], video_id: str) -> tuple[str, ...]:
        """Return units never evaluated on this video, for lazy backfill."""
        return tuple(
            unit_id
            for unit_id in unit_ids
            if self.get(unit_id, video_id).status is EvidenceStatus.UNKNOWN
        )
