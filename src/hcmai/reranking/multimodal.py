"""Bounded query-image reranking over existing retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.common.utils.image import load_image


ScoreBatch = Callable[[str, Sequence[Any]], Sequence[Any]]


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for the standalone bounded reranker."""

    batch_size: int = 8
    final_score_policy: str = "reranker"
    failure_policy: str = "original_order"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.final_score_policy != "reranker":
            raise ValueError("only the reranker final-score policy is supported")
        if self.failure_policy != "original_order":
            raise ValueError("only original-order failure fallback is supported")


def _finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _existing_score(candidate: RetrievalCandidate) -> float | None:
    visual = candidate.source_scores.get(RetrievalSource.VISUAL)
    for value in (candidate.final_score, candidate.fusion_score, visual):
        if _finite(value):
            return float(value)
    return None


def _replace(candidate: RetrievalCandidate, **updates: Any) -> RetrievalCandidate:
    values = candidate.model_dump(mode="python")
    values.update(updates)
    return RetrievalCandidate.model_validate(values)


def _fallback(
    candidate: RetrievalCandidate,
    category: str,
    message: str,
    *,
    candidate_level: bool,
) -> RetrievalCandidate:
    metadata = dict(candidate.metadata)
    metadata["reranker_fallback"] = {
        "category": category[:80],
        "message": (message.strip() or category)[:200],
    }
    updates: dict[str, Any] = {"metadata": metadata}
    if candidate_level:
        updates.update(reranker_score=None, final_score=_existing_score(candidate))
    return _replace(candidate, **updates)


class MultimodalReranker:
    """Score only supplied candidates using canonical frame images."""

    def __init__(
        self,
        frame_store: Any,
        config: RerankerConfig,
        score_batch: ScoreBatch | None = None,
        *,
        dataset_root: str | Path = ".",
    ) -> None:
        self.frame_store = frame_store
        self.config = config
        self.score_batch = score_batch
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    def _prepare(
        self, candidates: list[RetrievalCandidate]
    ) -> tuple[list[RetrievalCandidate], list[tuple[int, Any]]]:
        copies = [_replace(candidate) for candidate in candidates]
        prepared: list[tuple[int, Any]] = []
        for position, candidate in enumerate(copies):
            try:
                frame = self.frame_store.get(candidate.frame_id)
                path = Path(str(frame.image_path)).expanduser()
                image_path = path if path.is_absolute() else self.dataset_root / path
                image = load_image(image_path, mode="RGB")
            except Exception as error:
                copies[position] = _fallback(
                    candidate, type(error).__name__, str(error), candidate_level=True
                )
            else:
                prepared.append((position, image))
        return copies, prepared

    def _score(
        self, query: str, prepared: list[tuple[int, Any]]
    ) -> tuple[list[tuple[int, Any]] | None, tuple[str, str] | None]:
        if self.score_batch is None:
            return None, ("BackendUnavailable", "score_batch is required")
        scored: list[tuple[int, Any]] = []
        for start in range(0, len(prepared), self.config.batch_size):
            batch = prepared[start : start + self.config.batch_size]
            try:
                values = list(self.score_batch(query, [image for _, image in batch]))
                if len(values) != len(batch):
                    raise ValueError("score backend returned the wrong result count")
            except Exception as error:
                return None, (type(error).__name__, str(error))
            scored.extend((position, value) for (position, _), value in zip(batch, values))
        return scored, None

    @staticmethod
    def _ordered(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        def key(item: tuple[int, RetrievalCandidate]) -> tuple[float, int, str]:
            position, candidate = item
            score = candidate.final_score
            primary = -float(score) if _finite(score) else math.inf
            return primary, position, candidate.frame_id

        return [candidate for _, candidate in sorted(enumerate(candidates), key=key)]

    def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """Return score-enriched copies without changing candidate identity."""
        original = list(candidates)
        if not original:
            return []
        copies, prepared = self._prepare(original)
        try:
            scored, failure = self._score(query, prepared) if prepared else ([], None)
        finally:
            for _, image in prepared:
                image.close()
        if failure is not None:
            return [
                _fallback(candidate, *failure, candidate_level=False)
                for candidate in original
            ]
        for position, value in scored or []:
            if _finite(value):
                score = float(value)
                copies[position] = _replace(
                    copies[position], reranker_score=score, final_score=score
                )
            else:
                copies[position] = _fallback(
                    copies[position], "InvalidScore", repr(value), candidate_level=True
                )
        return self._ordered(copies)
