"""Registry for runnable baseline ranking methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from baseline.contracts import BASELINE_METHODS, BaselinePath, MethodName, MethodOptions
from baseline.methods.framewise import (
    DanteStyleUnaryDPMethod,
    GlobalQueryMethod,
    IndependentEventsMethod,
)
from baseline.methods.lattice import (
    UnaryBoundedOrderMethod,
    UnaryCandidateLatticeMethod,
)
from hcmai.retrieval.retriever.video_scores import VideoEventScores


class BaselineMethod(Protocol):
    """Common ranking boundary used by the baseline runner."""

    name: str

    def rank(
        self,
        videos: Sequence[VideoEventScores],
        *,
        top_k: int,
    ) -> list[BaselinePath]: ...


def create_method(name: MethodName, options: MethodOptions) -> BaselineMethod:
    """Construct one registered method with immutable numerical options."""

    if name == "global_query":
        return GlobalQueryMethod()
    if name == "independent_events":
        return IndependentEventsMethod()
    if name == "dante_style_unary_dp":
        return DanteStyleUnaryDPMethod(options)
    if name == "unary_candidate_lattice":
        return UnaryCandidateLatticeMethod(options)
    if name == "unary_bounded_order":
        return UnaryBoundedOrderMethod(options)
    valid = ", ".join(BASELINE_METHODS)
    raise ValueError(f"unsupported baseline method {name!r}; choose one of: {valid}")


__all__ = ["BaselineMethod", "create_method"]
