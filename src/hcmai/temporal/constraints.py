"""Validate event-scoped intervals and build pure temporal frame masks.

This module owns closed integer-millisecond arithmetic only. It intentionally
does not perform retrieval, model inference, corpus access, or identity joins.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

Interval = tuple[int, int]
DomainStatus = Literal["ready", "contradictory_conditions", "no_indexed_frames"]


@dataclass(frozen=True, slots=True)
class Conditions:
    """Per-event confirmation and rejection intervals within one query window."""

    window: Interval
    confirmed: tuple[Interval | None, ...]
    rejected: tuple[tuple[Interval, ...], ...]


def validate_interval(interval: Interval) -> None:
    """Validate a closed interval with non-negative integer endpoints."""
    try:
        if len(interval) != 2:
            raise ValueError("interval must contain exactly two endpoints")
        start_ms, end_ms = interval
    except (TypeError, ValueError):
        raise ValueError("interval must contain exactly two endpoints") from None
    if (
        isinstance(start_ms, bool)
        or isinstance(end_ms, bool)
        or not isinstance(start_ms, Integral)
        or not isinstance(end_ms, Integral)
        or start_ms < 0
        or start_ms > end_ms
    ):
        raise ValueError("interval endpoints must be non-negative integers in order")


def merge_intervals(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]:
    """Merge overlapping and integer-adjacent closed intervals."""
    validated = []
    for interval in intervals:
        validate_interval(interval)
        validated.append((int(interval[0]), int(interval[1])))
    if not validated:
        return ()

    merged = []
    for start_ms, end_ms in sorted(validated):
        if merged and start_ms <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))
        else:
            merged.append((start_ms, end_ms))
    return tuple(merged)


def remaining_intervals(
    domain: Interval, rejected: tuple[Interval, ...]
) -> tuple[Interval, ...]:
    """Subtract rejected closed intervals from a closed integer domain."""
    validate_interval(domain)
    current = [domain]
    for reject_start, reject_end in merge_intervals(rejected):
        next_current = []
        for start_ms, end_ms in current:
            if reject_end < start_ms or reject_start > end_ms:
                next_current.append((start_ms, end_ms))
                continue
            if start_ms < reject_start:
                next_current.append((start_ms, reject_start - 1))
            if reject_end < end_ms:
                next_current.append((reject_end + 1, end_ms))
        current = next_current
    return tuple(current)


def build_mask(
    timestamps_ms: np.ndarray, conditions: Conditions
) -> tuple[np.ndarray, DomainStatus]:
    """Build an event-by-frame mask from independent temporal conditions.

    A non-empty mathematical interval can still have no sampled timestamp;
    that case reports ``no_indexed_frames`` rather than contradiction.
    """
    if not isinstance(conditions, Conditions):
        raise ValueError("conditions must be a Conditions instance")
    validate_interval(conditions.window)
    event_count = len(conditions.confirmed)
    if event_count == 0 or len(conditions.rejected) != event_count:
        raise ValueError("conditions need equal, non-empty event sequences")
    if not isinstance(timestamps_ms, np.ndarray):
        raise ValueError("timestamps_ms must be a numpy array")
    if timestamps_ms.ndim != 1 or not np.issubdtype(timestamps_ms.dtype, np.integer):
        raise ValueError("timestamps_ms must be a one-dimensional integer array")
    if np.any(timestamps_ms < 0) or np.any(timestamps_ms[1:] < timestamps_ms[:-1]):
        raise ValueError("timestamps_ms must be non-negative and nondecreasing")

    domains = []
    for confirmation, rejected in zip(conditions.confirmed, conditions.rejected):
        if not isinstance(rejected, tuple):
            raise ValueError("each rejected event value must be a tuple")
        if confirmation is not None:
            validate_interval(confirmation)
            domain = (
                max(conditions.window[0], confirmation[0]),
                min(conditions.window[1], confirmation[1]),
            )
            if domain[0] > domain[1]:
                domains.append(())
                continue
        else:
            domain = conditions.window
        domains.append(remaining_intervals(domain, rejected))

    mask = np.zeros((event_count, timestamps_ms.size), dtype=bool)
    if any(not domain for domain in domains):
        # Contradiction outranks sampling availability across event rows.
        return mask, "contradictory_conditions"
    indexed = False
    for event_index, domain_parts in enumerate(domains):
        for start_ms, end_ms in domain_parts:
            mask[event_index] |= (timestamps_ms >= start_ms) & (timestamps_ms <= end_ms)
        indexed |= bool(mask[event_index].any())
    return mask, "ready" if indexed else "no_indexed_frames"
