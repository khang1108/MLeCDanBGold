"""Bounded query-image reranking over existing retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from hcmai.common.schemas import RetrievalCandidate
from hcmai.common.utils.image import load_image
from hcmai.common.utils.logging import get_logger
from hcmai.reranking.multimodal.config import RerankerConfig
from hcmai.reranking.multimodal.protocols import ScoreBatch

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
        try:
            for position, candidate in enumerate(copies):
                frame = self.frame_store.get(candidate.frame_id)
                path = Path(str(frame.image_path)).expanduser()
                image_path = path if path.is_absolute() else self.dataset_root / path
                prepared.append((position, load_image(image_path, mode="RGB")))
        except Exception:
            for _, image in prepared:
                image.close()
            raise
        return copies, prepared

    def _score(
        self, query: str, prepared: list[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        if self.score_batch is None:
            raise RuntimeError("score_batch is required")
        scored: list[tuple[int, Any]] = []
        for start in range(0, len(prepared), self.config.batch_size):
            batch = prepared[start : start + self.config.batch_size]
            batch_number = start // self.config.batch_size + 1
            batch_total = math.ceil(len(prepared) / self.config.batch_size)
            logger.info(
                "Reranker inference batch started batch=%d/%d images=%d",
                batch_number, batch_total, len(batch),
            )
            values = list(self.score_batch(query, [image for _, image in batch]))
            if len(values) != len(batch):
                raise ValueError("score backend returned the wrong result count")
            scored.extend((position, value) for (position, _), value in zip(batch, values))
            logger.info(
                "Reranker inference batch completed batch=%d/%d scores=%d",
                batch_number, batch_total, len(values),
            )
        return scored

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
            scored = self._score(query, prepared)
        finally:
            for _, image in prepared:
                image.close()
        for position, value in scored:
            if not _finite(value):
                raise ValueError(f"reranker returned invalid score: {value!r}")
            score = float(value)
            copies[position] = _replace(
                copies[position], reranker_score=score, final_score=score
            )
        ordered = self._ordered(copies)
        logger.info(
            "Reranking pipeline completed candidates=%d scored=%d",
            len(ordered), len(scored),
        )
        return ordered

def _finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _replace(candidate: RetrievalCandidate, **updates: Any) -> RetrievalCandidate:
    values = candidate.model_dump(mode="python")
    values.update(updates)
    return RetrievalCandidate.model_validate(values)
