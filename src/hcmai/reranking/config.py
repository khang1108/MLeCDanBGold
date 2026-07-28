from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class QwenRerankerConfig:
    """CPU-compatible configuration for the native Qwen3-VL reranker."""

    checkpoint: str = "Qwen/Qwen3-VL-Reranker-2B"
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "bfloat16"
    batch_size: int = 1
    max_length: int = 1024
    max_pixels: int = 262144
    instruction: str = (
        "Given a natural-language video-frame search query, determine whether "
        "the candidate image is relevant to the query."
    )

    def __post_init__(self) -> None:
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.batch_size < 1 or self.max_length < 1 or self.max_pixels < 4096:
            raise ValueError("batch, length, and pixel limits must be positive")
        if not self.checkpoint.strip() or not self.instruction.strip():
            raise ValueError("checkpoint and instruction must be non-empty")
