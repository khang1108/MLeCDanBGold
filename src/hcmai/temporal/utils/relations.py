"""Deterministic Vietnamese/English temporal-relation parsing and scoring."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from hcmai.common.schemas import (
    FrameEvidence,
    QueryUnit,
    TemporalConstraint,
    TemporalRelation,
)


def parse_temporal_constraints(units: list[QueryUnit]) -> list[TemporalConstraint]:
    """Attach only explicit relations; reveal order by itself emits nothing."""

    constraints: list[TemporalConstraint] = []
    for index, unit in enumerate(units):
        text = _normalize(unit.text)
        previous = units[index - 1] if index else None
        if _contains(
            text,
            ("cuối cùng", "ở cuối cảnh", "at the end", "finally"),
        ):
            constraints.append(TemporalConstraint(
                relation=TemporalRelation.AT_END,
                subject_unit_id=unit.unit_id,
                reason="explicit_at_end",
            ))
            continue
        if previous is None:
            continue
        # Reveal order only identifies the units referenced by an explicit
        # trigger. Without a trigger, adjacent hints produce no constraint.
        if _contains(
            text,
            (
                "đồng thời",
                "cùng lúc",
                "trong lúc",
                "simultaneously",
                "at the same time",
            ),
        ):
            constraints.append(TemporalConstraint(
                relation=TemporalRelation.OVERLAP,
                subject_unit_id=previous.unit_id,
                object_unit_id=unit.unit_id,
                reason="explicit_overlap",
            ))
        elif _contains(
            text,
            (
                "sau đó",
                "rồi",
                "then",
            ),
        ):
            constraints.append(TemporalConstraint(
                relation=TemporalRelation.BEFORE,
                subject_unit_id=previous.unit_id,
                object_unit_id=unit.unit_id,
                reason="explicit_before_after",
            ))
    return constraints


def relation_satisfaction(
    constraints: list[TemporalConstraint],
    evidence: tuple[FrameEvidence, ...],
    *,
    near_ms: int,
) -> tuple[float | None, tuple[str, ...]]:
    """Score explicit soft constraints against timestamps in one scene."""

    if not constraints:
        return None, ("no_explicit_relation",)
    timestamps: dict[str, list[int]] = defaultdict(list)
    for item in evidence:
        for unit_id in item.unit_scores:
            timestamps[unit_id].append(item.frame.timestamp_ms)
    scores: list[float] = []
    labels: list[str] = []
    all_times = [value for values in timestamps.values() for value in values]
    for constraint in constraints:
        subject = timestamps.get(constraint.subject_unit_id, [])
        object_values = timestamps.get(constraint.object_unit_id or "", [])
        if not subject or (constraint.object_unit_id and not object_values):
            labels.append(f"relation_unknown:{constraint.reason}")
            continue
        satisfied = False
        if constraint.relation is TemporalRelation.BEFORE:
            satisfied = any(left <= right for left in subject for right in object_values)
        elif constraint.relation is TemporalRelation.AFTER:
            satisfied = any(left >= right for left in subject for right in object_values)
        elif constraint.relation is TemporalRelation.OVERLAP:
            satisfied = any(
                abs(left - right) <= near_ms
                for left in subject
                for right in object_values
            )
        elif constraint.relation is TemporalRelation.NEAR:
            satisfied = any(
                abs(left - right) <= near_ms
                for left in subject
                for right in object_values
            )
        elif constraint.relation is TemporalRelation.AT_END:
            satisfied = bool(all_times) and max(subject) == max(all_times)
        scores.append(1.0 if satisfied else 0.0)
        outcome = "satisfied" if satisfied else "violated"
        labels.append(f"relation_{outcome}:{constraint.reason}")
    return (sum(scores) / len(scores) if scores else None), tuple(labels)


def _normalize(value: str) -> str:
    """Normalize relation text conservatively for deterministic matching."""

    normalized = unicodedata.normalize("NFC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _contains(value: str, phrases: tuple[str, ...]) -> bool:
    """Return whether normalized text contains any explicit trigger phrase."""

    return any(phrase in value for phrase in phrases)
