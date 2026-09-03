"""Immutable evidence components and bundles for temporal scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np


@dataclass(frozen=True, slots=True)
class TemporalScoreComponent:
    name: str
    raw_scores: np.ndarray
    coverage: np.ndarray | None = None

    def __post_init__(self) -> None:
        scores = np.asarray(self.raw_scores, dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError("component scores must be two-dimensional")
        if not np.all(np.isfinite(scores)):
            raise ValueError("component scores must contain only finite values")
        if self.coverage is not None:
            coverage = np.asarray(self.coverage, dtype=bool)
            if coverage.shape != (scores.shape[1],):
                raise ValueError("component coverage must match frame count")
            object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "raw_scores", scores)


@dataclass(frozen=True, slots=True)
class TemporalScoreBundle:
    components: Mapping[str, TemporalScoreComponent]

    def __post_init__(self) -> None:
        copied = dict(self.components)
        if not copied:
            raise ValueError("temporal score bundle must contain at least one component")
        shapes = {component.raw_scores.shape for component in copied.values()}
        if len(shapes) != 1:
            raise ValueError("all temporal components must have the same score shape")
        for key, component in copied.items():
            if key != component.name:
                raise ValueError("component mapping key must match component name")
        object.__setattr__(self, "components", MappingProxyType(copied))

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.components.values())).raw_scores.shape
