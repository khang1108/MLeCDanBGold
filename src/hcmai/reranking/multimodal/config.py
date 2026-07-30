from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for the standalone bounded reranker."""

    batch_size: int = 8
    final_score_policy: str = "reranker"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.final_score_policy != "reranker":
            raise ValueError("only the reranker final-score policy is supported")
