from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Final

from hcmai.temporal.models import QueryUnit, RelationType, TemporalConstraint


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """Query units after folding a cumulative hint snapshot into previous ones."""

    units: tuple[QueryUnit, ...]
    delta: QueryUnit | None = None
    warnings: tuple[str, ...] = ()


def diff_snapshot(
    snapshot: str,
    previous_snapshot: str = "",
    units: tuple[QueryUnit, ...] = (),
) -> SnapshotDiff:
    """Diff a cumulative snapshot against the previous one, deterministically."""
    new_sentences = _sentences(snapshot)
    old_sentences = _sentences(previous_snapshot)
    if not new_sentences:
        return SnapshotDiff(units)

    new_keys = [_key(sentence) for sentence in new_sentences]
    old_keys = [_key(sentence) for sentence in old_sentences]
    if new_keys == old_keys:
        return SnapshotDiff(units)

    old_key_set = set(old_keys)
    if new_keys[: len(old_keys)] == old_keys:
        added = new_sentences[len(old_keys) :]
    elif old_key_set <= set(new_keys):
        added = [
            sentence
            for sentence, key in zip(new_sentences, new_keys)
            if key not in old_key_set
        ]
    else:
        unit = QueryUnit(unit_id="unit-0", text=" ".join(new_sentences), reveal_index=0)
        return SnapshotDiff(
            units=(unit,),
            delta=unit,
            warnings=("snapshot is not cumulative; previous units were replaced",),
        )

    if not added:
        return SnapshotDiff(units)

    delta = QueryUnit(
        unit_id=f"unit-{len(units)}",
        text=" ".join(added),
        reveal_index=len(units),
    )
    return SnapshotDiff(units=(*units, delta), delta=delta)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text)
    stripped = (" ".join(part.split()) for part in parts)
    return [part for part in stripped if part.strip(".,;:!?…")]


def _key(sentence: str) -> str:
    return sentence.casefold().strip(".,;:!?…")


_FORWARD_MARKERS: Final = (
    "sau đó",
    "rồi",
    "cuối cùng",
    "then",
    "afterwards",
    "finally",
    "sau khi",
    "after",
)
_BACKWARD_MARKERS: Final = ("trước đó", "trước khi", "before that", "previously")
_SIMULTANEOUS_MARKERS: Final = (
    "đồng thời",
    "cùng lúc",
    "meanwhile",
    "at the same time",
    "simultaneously",
)

_MARKER_PATTERNS: Final = {
    marker: re.compile(rf"\b{re.escape(marker)}\b")
    for marker in (*_FORWARD_MARKERS, *_BACKWARD_MARKERS, *_SIMULTANEOUS_MARKERS)
}


@dataclass(frozen=True, slots=True)
class TemporalParseResult:
    """Immutable outcome of a rule-based temporal relation parse."""

    constraints: tuple[TemporalConstraint, ...] = ()
    warnings: tuple[str, ...] = ()


class RuleTemporalRelationParser:
    """Lexical baseline reading temporal relations between adjacent query units.

    A relation is emitted only when a later unit's own text explicitly carries a
    marker that points at the immediately previous unit. Plain text never yields
    a constraint, regardless of reveal order. Uncertain parses (conflicting
    directions, or a marker with no previous unit) yield no constraint plus a
    deterministic contextual warning.
    """

    def parse(self, units: tuple[QueryUnit, ...]) -> TemporalParseResult:
        constraints: list[TemporalConstraint] = []
        warnings: list[str] = []
        orphans = (
            (
                *_matched_markers(units[0].text, _FORWARD_MARKERS),
                *_matched_markers(units[0].text, _BACKWARD_MARKERS),
                *_matched_markers(units[0].text, _SIMULTANEOUS_MARKERS),
            )
            if units
            else ()
        )
        if orphans:
            warnings.append(
                f"temporal marker {orphans[0]!r} in unit {units[0].unit_id!r} "
                "has no previous unit"
            )
        for previous, current in zip(units, units[1:]):
            pair_constraints, pair_warnings = _parse_pair(previous, current)
            constraints.extend(pair_constraints)
            warnings.extend(pair_warnings)
        return TemporalParseResult(
            constraints=tuple(constraints),
            warnings=tuple(warnings),
        )


def _parse_pair(
    previous: QueryUnit,
    current: QueryUnit,
) -> tuple[list[TemporalConstraint], list[str]]:
    forward = _matched_markers(current.text, _FORWARD_MARKERS)
    backward = _matched_markers(current.text, _BACKWARD_MARKERS)
    simultaneous = _matched_markers(current.text, _SIMULTANEOUS_MARKERS)
    categories = tuple(
        name
        for name, matched in (
            ("forward", forward),
            ("backward", backward),
            ("simultaneous", simultaneous),
        )
        if matched
    )
    if len(categories) > 1:
        return [], [
            f"conflicting temporal markers in unit {current.unit_id!r}: "
            f"{', '.join(categories)}"
        ]
    if forward:
        return [
            TemporalConstraint(
                previous.unit_id, RelationType.BEFORE, current.unit_id
            )
        ], []
    if backward:
        return [
            TemporalConstraint(
                current.unit_id, RelationType.BEFORE, previous.unit_id
            )
        ], []
    if simultaneous:
        return [
            TemporalConstraint(
                previous.unit_id, RelationType.OVERLAPS, current.unit_id
            )
        ], []
    return [], []


def _matched_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    folded = text.casefold()
    return tuple(marker for marker in markers if _MARKER_PATTERNS[marker].search(folded))
