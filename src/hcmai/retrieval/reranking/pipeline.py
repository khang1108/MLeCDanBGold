"""Public reranking service facade."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from hcmai.common.schemas import RetrievalCandidate
from hcmai.common.utils.image import load_image
from hcmai.common.utils.logging import get_logger
from hcmai.data.pipeline import DataService
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
        data: DataService,
        config: RerankerConfig,
        adapter: RerankingAdapter,
        *,
        dataset_root: str | Path = ".",
    ) -> None:
        self.data = data
        self.config = config
        self.adapter = adapter
        self.dataset_root = Path(dataset_root).expanduser().resolve()

    @classmethod
    def remote(
        cls,
        data: DataService,
        config: RerankerConfig,
        client: Any,
        *,
        dataset_root: str | Path = ".",
    ) -> RerankingService:
        from hcmai.retrieval.reranking.adapters.remote import RemoteAdapter

        return cls(
            data,
            config,
            RemoteAdapter(client),
            dataset_root=dataset_root,
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
                    frame = self.data.get_frame(candidate.frame_id)
                    asset_reference = getattr(frame, "image_path", asset_reference)
                    if isinstance(self.data, DataService):
                        image_path = self.data.resolve_frame_asset(frame)
                    else:
                        path = Path(str(frame.image_path)).expanduser()
                        image_path = path if path.is_absolute() else self.dataset_root / path
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
    values = candidate.model_dump(mode="python")
    values.update(updates)
    return RetrievalCandidate.model_validate(values)


def _classified_backend_error(error: Exception) -> RerankingError:
    raw_category = getattr(error, "category", None)
    category = str(getattr(raw_category, "value", raw_category) or "")
    if category in {"timeout", "deadline_exceeded"}:
        return RerankerTimeoutError()
    if category in {"client_error", "invalid_response"}:
        return RerankerContractError()
    return RerankerUnavailableError()
