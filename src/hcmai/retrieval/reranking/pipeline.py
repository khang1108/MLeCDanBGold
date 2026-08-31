"""Public reranking service facade."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from numbers import Real
from typing import Any

from hcmai.retrieval.models import RetrievalCandidate
from hcmai.common.utils.image import load_image
from hcmai.common.utils.logging import get_logger
from hcmai.corpus import Corpus
from hcmai.retrieval.reranking.config import QwenRerankerConfig, RerankerConfig
from hcmai.retrieval.reranking.models import HostedRerankingAdapter, RerankingAdapter

__all__ = [
    "QwenRerankerConfig",
    "RerankerConfig",
    "HostedRerankingAdapter",
    "RerankingAdapter",
    "RerankingError",
    "RerankerContractError",
    "RerankerInvalidScoreError",
    "RerankerTimeoutError",
    "RerankerUnavailableError",
    "RerankingService",
]

logger = get_logger(__name__)


class RerankingError(RuntimeError):
    """Safe categorized reranking failure exposed to orchestration."""

    def __init__(self, category: str) -> None:
        super().__init__(f"reranking failed ({category})")
        self.category = category


class RerankerUnavailableError(RerankingError):
    def __init__(self, category: str = "unavailable") -> None:
        super().__init__(category)


class RerankerTimeoutError(RerankingError):
    def __init__(self) -> None:
        super().__init__("timeout")


class RerankerContractError(RerankingError):
    def __init__(self) -> None:
        super().__init__("contract_error")


class RerankerInvalidScoreError(RerankingError):
    def __init__(self) -> None:
        super().__init__("invalid_score")


class RerankingService:
    """Prepare canonical frame images and delegate bounded model scoring."""

    def __init__(
        self,
        corpus: Corpus,
        config: RerankerConfig,
        adapter: RerankingAdapter,
    ) -> None:
        self.corpus = corpus
        self.config = config
        self.adapter = adapter

    @classmethod
    def remote(
        cls,
        corpus: Corpus,
        config: RerankerConfig,
        client: Any,
    ) -> RerankingService:
        from hcmai.retrieval.reranking.adapters.remote import RemoteAdapter

        return cls(
            corpus,
            config,
            RemoteAdapter(client),
        )

    @staticmethod
    def create_qwen_adapter(
        config: QwenRerankerConfig,
    ) -> HostedRerankingAdapter:
        """Create the configured local scorer behind the public boundary."""

        from hcmai.retrieval.reranking.adapters.qwen import QwenAdapter

        return QwenAdapter(config)

    def rerank(
        self, query: str, candidates: Sequence[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        """Return score-enriched copies without changing candidate identity."""
        original = list(candidates)
        if not original:
            return []
        try:
            copies, prepared = self._prepare(original)
        except FileNotFoundError as error:
            raise RerankerUnavailableError("frame_asset_missing") from error
        except (OSError, KeyError, RuntimeError) as error:
            raise RerankerUnavailableError("image_load_failure") from error
        logger.info("Reranker images prepared loaded=%d", len(prepared))
        try:
            try:
                scored = self._score(query, prepared)
            except RerankingError:
                raise
            except TimeoutError as error:
                raise RerankerTimeoutError() from error
            except Exception as error:
                raise _classified_backend_error(error) from error
        finally:
            for _, image in prepared:
                image.close()
        for position, value in scored:
            if not _finite(value):
                raise RerankerInvalidScoreError()
            score = float(value)
            copies[position] = _replace(
                copies[position], reranker_score=score, final_score=score
            )
        return self._ordered(copies)

    def _prepare(
        self, candidates: list[RetrievalCandidate]
    ) -> tuple[list[RetrievalCandidate], list[tuple[int, Any]]]:
        copies = [_replace(candidate) for candidate in candidates]
        prepared: list[tuple[int, Any]] = []
        try:
            for position, candidate in enumerate(copies):
                asset_reference: object = "<unresolved>"
                try:
                    image_path = self.corpus.image_path(candidate.frame_id)
                    asset_reference = image_path
                    prepared.append((position, load_image(image_path, mode="RGB")))
                except Exception as error:
                    # Keep the public fallback category stable, but retain the
                    # first failing candidate and root cause in server logs.
                    logger.warning(
                        "Reranker image preparation failed frame_id=%s asset=%s "
                        "error_type=%s error=%s",
                        candidate.frame_id,
                        asset_reference,
                        type(error).__name__,
                        error,
                    )
                    raise
        except Exception:
            for _, image in prepared:
                image.close()
            raise
        return copies, prepared

    def _score(
        self, query: str, prepared: list[tuple[int, Any]]
    ) -> list[tuple[int, Any]]:
        scored: list[tuple[int, Any]] = []
        for start in range(0, len(prepared), self.config.batch_size):
            batch = prepared[start : start + self.config.batch_size]
            values = list(self.adapter.score(query, [image for _, image in batch]))
            if len(values) != len(batch):
                raise RerankerContractError()
            scored.extend((position, value) for (position, _), value in zip(batch, values))
        return scored

    @staticmethod
    def _ordered(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        def key(item: tuple[int, RetrievalCandidate]) -> tuple[float, int, str]:
            position, candidate = item
            score = candidate.final_score
            primary = -float(score) if score is not None and _finite(score) else math.inf
            return primary, position, candidate.frame_id

        return [candidate for _, candidate in sorted(enumerate(candidates), key=key)]


def _finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _replace(candidate: RetrievalCandidate, **updates: Any) -> RetrievalCandidate:
    """Return an immutable candidate with explicitly updated ranking fields."""

    return replace(candidate, **updates)


def _classified_backend_error(error: Exception) -> RerankingError:
    raw_category = getattr(error, "category", None)
    category = str(getattr(raw_category, "value", raw_category) or "")
    if category in {"timeout", "deadline_exceeded"}:
        return RerankerTimeoutError()
    if category in {"client_error", "invalid_response"}:
        return RerankerContractError()
    return RerankerUnavailableError()
