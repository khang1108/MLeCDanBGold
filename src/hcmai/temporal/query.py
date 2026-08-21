"""Deterministic cumulative-snapshot normalization and differencing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class SnapshotDiffMode(str, Enum):
    """Describe how the current cumulative snapshot differs from the prior one."""

    FIRST = "first"
    APPEND = "append"
    NO_CHANGE = "no_change"
    REPLACEMENT = "replacement"


@dataclass(frozen=True)
class SnapshotDiffResult:
    """Return normalized snapshots and a safe incremental query-unit suffix."""

    normalized_previous: str
    normalized_current: str
    delta_text: str | None
    changed: bool
    mode: SnapshotDiffMode


def normalize_snapshot(value: str) -> str:
    """Apply one conservative normalization shared by KIS and VQA."""

    normalized = unicodedata.normalize("NFC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", normalized)
    return normalized


def diff_snapshot(previous: str | None, current: str) -> SnapshotDiffResult:
    """Return only a safe appended suffix; never infer a semantic rewrite."""

    current_normalized = normalize_snapshot(current)
    if not current_normalized:
        raise ValueError("current snapshot must not be empty")
    previous_normalized = normalize_snapshot(previous or "")
    if not previous_normalized:
        return SnapshotDiffResult(
            previous_normalized,
            current_normalized,
            current_normalized,
            True,
            SnapshotDiffMode.FIRST,
        )
    if _semantic_key(previous_normalized) == _semantic_key(current_normalized):
        return SnapshotDiffResult(
            previous_normalized,
            current_normalized,
            None,
            False,
            SnapshotDiffMode.NO_CHANGE,
        )
    if current_normalized.startswith(previous_normalized):
        suffix = current_normalized[len(previous_normalized) :].strip()
        suffix = suffix.lstrip(".,;:!?-–— ").strip()
        if suffix and _semantic_key(suffix):
            return SnapshotDiffResult(
                previous_normalized,
                current_normalized,
                suffix,
                True,
                SnapshotDiffMode.APPEND,
            )
    return SnapshotDiffResult(
        previous_normalized,
        current_normalized,
        None,
        False,
        SnapshotDiffMode.REPLACEMENT,
    )


def _semantic_key(value: str) -> str:
    """Build a comparison key that ignores whitespace and punctuation changes."""

    return "".join(
        character.casefold()
        for character in value
        if not unicodedata.category(character).startswith(("P", "Z"))
    )
