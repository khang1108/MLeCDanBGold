"""Bounded query-image reranking over existing retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from hcmai.common.schemas import RetrievalCandidate, RetrievalSource
from hcmai.common.utils.image import load_image
from hcmai.common.utils.logging import get_logger
from hcmai.reranking.config import RerankerConfig
from hcmai.reranking.protocols import ScoreBatch

logger = get_logger(__name__)


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
            batch_number = start // self.config.batch_size + 1
            batch_total = math.ceil(len(prepared) / self.config.batch_size)
            logger.info(
                "Reranker inference batch started batch=%d/%d images=%d",
                batch_number, batch_total, len(batch),
            )
            try:
                values = list(self.score_batch(query, [image for _, image in batch]))
                if len(values) != len(batch):
                    raise ValueError("score backend returned the wrong result count")
            except Exception as error:
                logger.warning(
                    "Reranker inference failed batch=%d/%d error=%s: %s",
                    batch_number, batch_total, type(error).__name__,
                    _bounded_message(error),
                )
                return None, (type(error).__name__, str(error))
            scored.extend((position, value) for (position, _), value in zip(batch, values))
            logger.info(
                "Reranker inference batch completed batch=%d/%d scores=%d",
                batch_number, batch_total, len(values),
            )
        return scored, None

    @staticmethod
    def _ordered(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        def key(item: tuple[int, RetrievalCandidate]) -> tuple[float, int, str]:
            position, candidate = item
            score = candidate.final_score
            primary = -float(score) if score is not None and _finite(score) else math.inf
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
        logger.info(
            "Reranker images prepared loaded=%d missing=%d",
            len(prepared), len(original) - len(prepared),
        )
        try:
            scored, failure = self._score(query, prepared) if prepared else ([], None)
        finally:
            for _, image in prepared:
                image.close()
        if failure is not None:
            logger.warning(
                "Reranking preserved dense order for all candidates "
                "category=%s message=%s",
                failure[0], _bounded_message(failure[1]),
            )
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
        ordered = self._ordered(copies)
        fallback_count = sum(
            "reranker_fallback" in (candidate.metadata or {})
            for candidate in ordered
        )
        logger.info(
            "Reranking pipeline completed candidates=%d scored=%d fallbacks=%d",
            len(ordered), len(scored or []), fallback_count,
        )
        return ordered

def _finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _bounded_message(value: object, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    return compact[:limit] or type(value).__name__


def _existing_score(candidate: RetrievalCandidate) -> float | None:
    visual = candidate.source_scores.get(RetrievalSource.VISUAL)
    for value in (candidate.final_score, candidate.fusion_score, visual):
        if value is not None and _finite(value):
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
