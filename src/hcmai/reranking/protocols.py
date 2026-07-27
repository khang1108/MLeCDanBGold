"""Interfaces implemented or consumed by reranking backends."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

ScoreBatch = Callable[[str, Sequence[Any]], Sequence[Any]]
