"""Deterministically convert a KIS or TRAKE query into ordered events.

This module owns only query segmentation and plan construction. It does not
infer event meaning with an LLM, retrieve frames, or perform temporal
alignment, which keeps the visual-only baseline reproducible.
"""

from __future__ import annotations

import re

from hcmai.common.schemas import AlignmentEvent, AlignmentPlan, SearchFilters

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def build_alignment_plan(
    query: str,
    events: list[str] | None = None,
    filters: SearchFilters | None = None,
) -> AlignmentPlan:
    """Build one deterministic ordered plan from explicit events or query text.

    Explicit events preserve caller order. Otherwise, nonempty lines take
    precedence over sentence boundaries, and a single remaining query becomes
    one event. Whitespace is normalized without semantic rewriting.
    """

    if events is not None:
        parts = _normalize_parts(events, strip_terminal_punctuation=False)
    else:
        lines = _normalize_parts(query.splitlines(), strip_terminal_punctuation=False)
        if len(lines) >= 2:
            parts = lines
        else:
            sentences = _normalize_parts(
                _SENTENCE_BOUNDARY.split(query),
                strip_terminal_punctuation=True,
            )
            parts = sentences if len(sentences) >= 2 else _normalize_parts(
                [query],
                strip_terminal_punctuation=False,
            )

    if not parts:
        raise ValueError("alignment query must contain at least one event")

    return AlignmentPlan(
        events=tuple(
            AlignmentEvent(event_id=f"e{index}", text=text, order=index)
            for index, text in enumerate(parts)
        ),
        filters=filters,
    )


def _normalize_parts(
    values: list[str],
    *,
    strip_terminal_punctuation: bool,
) -> list[str]:
    """Normalize whitespace and remove only sentence delimiters when splitting."""

    parts = [" ".join(value.split()) for value in values]
    if strip_terminal_punctuation:
        parts = [part.rstrip(".!?").rstrip() for part in parts]
    return [part for part in parts if part]
