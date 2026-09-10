"""Stable values shared by baseline query loaders, rankers, and CLIs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypeAlias


MethodName: TypeAlias = Literal[
    "global_query",
    "independent_events",
    "dante_style_unary_dp",
    "unary_candidate_lattice",
    "unary_bounded_order",
]
QueryKind: TypeAlias = Literal["kis", "qa", "trake"]

BASELINE_METHODS: tuple[MethodName, ...] = (
    "global_query",
    "independent_events",
    "dante_style_unary_dp",
    "unary_candidate_lattice",
    "unary_bounded_order",
)
QUERY_KINDS: tuple[QueryKind, ...] = ("kis", "qa", "trake")


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One deterministic benchmark query and its event decomposition."""

    query_file: str
    kind: QueryKind
    raw_query: str
    events: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query_file.strip():
            raise ValueError("query_file must be non-empty")
        if self.kind not in QUERY_KINDS:
            raise ValueError(f"unsupported query kind: {self.kind}")
        if not self.raw_query.strip():
            raise ValueError("raw_query must be non-empty")
        if not self.events or any(not event.strip() for event in self.events):
            raise ValueError("events must contain non-empty text")


@dataclass(frozen=True, slots=True)
class BaselinePath:
    """One ranked video and its canonical frame assignments."""

    video_id: str
    score: float
    frame_ids: tuple[str, ...]
    frame_idxs: tuple[int, ...]
    timestamps_ms: tuple[int, ...]
    chronological_event_order: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise ValueError("video_id must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.frame_ids:
            raise ValueError("frame arrays must not be empty")
        if not (
            len(self.frame_ids)
            == len(self.frame_idxs)
            == len(self.timestamps_ms)
        ):
            raise ValueError("frame arrays must have equal lengths")
        if any(not frame_id.strip() for frame_id in self.frame_ids):
            raise ValueError("frame_ids must be non-empty")
        if self.chronological_event_order is not None and tuple(
            sorted(self.chronological_event_order)
        ) != tuple(range(len(self.frame_ids))):
            raise ValueError(
                "chronological_event_order must be a permutation of frame assignments"
            )


@dataclass(frozen=True, slots=True)
class MethodOptions:
    """Shared numerical controls for the five runnable methods."""

    lambda_gap: float = 1e-5
    event_power: float = 1.0
    cluster_delta: float = 0.0
    candidates_per_event: int = 32
    candidate_min_separation_ms: int = 0
    max_adjacent_swaps: int = 1
    max_order_candidates: int = 64
    swap_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_gap) or self.lambda_gap < 0:
            raise ValueError("lambda_gap must be finite and non-negative")
        if not math.isfinite(self.event_power) or self.event_power <= 0:
            raise ValueError("event_power must be finite and greater than zero")
        if not math.isfinite(self.cluster_delta) or self.cluster_delta < 0:
            raise ValueError("cluster_delta must be finite and non-negative")
        if self.candidates_per_event <= 0:
            raise ValueError("candidates_per_event must be greater than zero")
        if self.candidate_min_separation_ms < 0:
            raise ValueError("candidate_min_separation_ms must be non-negative")
        if self.max_adjacent_swaps < 0:
            raise ValueError("max_adjacent_swaps must be non-negative")
        if self.max_order_candidates <= 0:
            raise ValueError("max_order_candidates must be greater than zero")
        if not math.isfinite(self.swap_penalty) or self.swap_penalty < 0:
            raise ValueError("swap_penalty must be finite and non-negative")
