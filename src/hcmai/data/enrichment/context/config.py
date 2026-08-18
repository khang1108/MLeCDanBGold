"""Frozen configuration for deterministic FrameContext V1 serialization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameContextConfig:
    """Define the complete, dependency-relevant FrameContext V1 policy."""

    context_version: str = "frame-context-v1"
    caption_token_budget: int = 80
    ocr_token_budget: int = 80
    object_token_budget: int = 40
    min_ocr_quality: float = 0.5

    def __post_init__(self) -> None:
        """Reject policies that cannot produce a valid deterministic artifact."""

        if (
            not isinstance(self.context_version, str)
            or not self.context_version.strip()
        ):
            raise ValueError("context_version must not be empty")
        for name in (
            "caption_token_budget",
            "ocr_token_budget",
            "object_token_budget",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.min_ocr_quality, (int, float)) or isinstance(
            self.min_ocr_quality, bool
        ):
            raise ValueError("min_ocr_quality must be numeric")
        if not 0.0 <= float(self.min_ocr_quality) <= 1.0:
            raise ValueError("min_ocr_quality must be in [0, 1]")
