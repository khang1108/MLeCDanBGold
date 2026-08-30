"""Deterministically split a KIS query into ordered event text.

This module owns only backend query segmentation for the baseline temporal
search flow. It does not validate TRAKE requests, infer event meaning with an
LLM, retrieve frames, or perform temporal alignment.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_query_events(query: str) -> tuple[str, ...]:
    """Split one query into deterministic ordered event text.

    Nonempty lines take precedence over sentence boundaries, and a single
    remaining query becomes one event. Whitespace is normalized without
    semantic rewriting. The return value is intentionally a plain tuple so the
    baseline has no alignment DTO.
    """

    lines = _normalize_parts(query.splitlines(), strip_terminal_punctuation=False)
    if len(lines) >= 2:
        parts = lines
    else:
        sentences = _normalize_parts(
            _SENTENCE_BOUNDARY.split(query),
            strip_terminal_punctuation=True,
        )
        parts = (
            sentences
            if len(sentences) >= 2
            else _normalize_parts(
                [query],
                strip_terminal_punctuation=False,
            )
        )

    if not parts:
        raise ValueError("alignment query must contain at least one event")

    return tuple(parts)


def _normalize_parts(
    values: Sequence[str],
    *,
    strip_terminal_punctuation: bool,
) -> list[str]:
    """Normalize whitespace and remove only sentence delimiters when splitting."""

    parts = [" ".join(value.split()) for value in values]
    if strip_terminal_punctuation:
        parts = [part.rstrip(".!?").rstrip() for part in parts]
    return [part for part in parts if part]
