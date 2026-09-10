"""Deterministic discovery and parsing of competition-style query files."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
import re
from typing import cast

from baseline.contracts import QUERY_KINDS, QueryCase, QueryKind
from hcmai.temporal.planner import plan_query_events


_TRAKE_EVENT = re.compile(
    r"^(?:E\d+|Cảnh\s*\d+)\s*:?\s*(.+)$",
    re.IGNORECASE,
)
_SUFFIX_KIND: tuple[tuple[str, QueryKind], ...] = (
    ("-kis.txt", "kis"),
    ("-qa.txt", "qa"),
    ("-trake.txt", "trake"),
)


def load_query_cases(
    root: Path,
    splits: Sequence[str],
    kinds: Collection[str],
    max_queries: int | None,
) -> list[QueryCase]:
    """Return sorted selected query cases without reading ground truth."""

    normalized_splits = tuple(sorted({split.strip() for split in splits if split.strip()}))
    if not normalized_splits:
        raise ValueError("at least one split must be selected")

    selected_kinds = set(kinds)
    unsupported = selected_kinds.difference(QUERY_KINDS)
    if unsupported:
        raise ValueError(f"unsupported query kinds: {', '.join(sorted(unsupported))}")
    if not selected_kinds:
        raise ValueError("at least one query kind must be selected")
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be greater than zero")

    paths = sorted(
        path
        for split in normalized_splits
        for path in (root / split).glob("*.txt")
        if (_kind_from_path(path) in selected_kinds)
    )
    if max_queries is not None:
        paths = paths[:max_queries]
    return [_parse_query(path) for path in paths]


def _kind_from_path(path: Path) -> QueryKind | None:
    name = path.name.lower()
    for suffix, kind in _SUFFIX_KIND:
        if name.endswith(suffix):
            return kind
    return None


def _parse_query(path: Path) -> QueryCase:
    kind = _kind_from_path(path)
    if kind is None:
        raise ValueError(f"{path}: unsupported query filename")
    raw_query = path.read_text(encoding="utf-8-sig").strip()
    try:
        events = _parse_trake_events(raw_query) if kind == "trake" else plan_query_events(raw_query)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return QueryCase(
        query_file=str(path),
        kind=cast(QueryKind, kind),
        raw_query=raw_query,
        events=events,
    )


def _parse_trake_events(query: str) -> tuple[str, ...]:
    events = tuple(
        match.group(1).strip()
        for line in query.splitlines()
        if (match := _TRAKE_EVENT.match(line.strip()))
    )
    if not events:
        raise ValueError("no TRAKE events found")
    return events
