"""Common-contract adapter over the existing TRAKE monotonic DP alignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hcmai.retriever.video_scores import VideoEventScores
from hcmai.temporal.alignment.base import AlignmentResult
from hcmai.temporal.models import QueryUnit, RelationType, TemporalConstraint
from hcmai.temporal.plan import TemporalQueryPlan
from hcmai.trake import TrakePath, rank_paths


def plan_from_events(events: Sequence[str]) -> TemporalQueryPlan:
    """Chain verbatim caller-ordered events into query units with hard BEFORE links."""
    units = tuple(
        QueryUnit(unit_id=f"event-{index}", text=text, reveal_index=index)
        for index, text in enumerate(events)
    )
    return TemporalQueryPlan(
        units=units,
        constraints=tuple(
            TemporalConstraint(left.unit_id, RelationType.BEFORE, right.unit_id)
            for left, right in zip(units, units[1:])
        ),
    )


def ordered_events(plan: TemporalQueryPlan) -> tuple[str, ...]:
    """Return plan unit texts in reveal order, the legacy TRAKE event list."""
    units = sorted(plan.units, key=lambda unit: (unit.reveal_index, unit.unit_id))
    return tuple(unit.text for unit in units)


@dataclass(frozen=True, slots=True)
class MonotonicDPAligner:
    """Rank ordered-event paths through the unchanged TRAKE DP."""

    lambda_gap: float = 1e-5
    max_rows: int = 100
    event_power: float = 1.0
    cluster_delta: float = 0.0

    def align(
        self,
        plan: TemporalQueryPlan,
        evidence: Sequence[VideoEventScores],
    ) -> AlignmentResult[TrakePath]:
        event_count = len(plan.units)
        for video in evidence:
            if len(video.scores) != event_count:
                raise ValueError(
                    f"video {video.video_id} scores {len(video.scores)} events, "
                    f"plan has {event_count}"
                )
        paths = rank_paths(
            evidence,
            self.lambda_gap,
            self.max_rows,
            self.event_power,
            self.cluster_delta,
        )
        return AlignmentResult(candidates=tuple(paths))
