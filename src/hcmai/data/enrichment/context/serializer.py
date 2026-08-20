"""Serialize specialist text into the frozen FrameContext V1 representation.

This module performs deterministic string transforms only. It does not run or
import inference, retrieval, visual, speech, or transcript components.
"""

from __future__ import annotations

from .config import FrameContextConfig


def _truncate_whitespace_tokens(text: str, limit: int) -> str:
    """Collapse whitespace and retain at most ``limit`` whitespace tokens."""

    return " ".join(text.split()[:limit])


def serialize_frame_context(
    *,
    caption: str | None,
    ocr: str | None,
    objects: str | None,
    config: FrameContextConfig,
) -> str | None:
    """Serialize non-empty Caption, OCR, and Object text in frozen V1 order."""

    sections: list[str] = []
    for heading, text, budget in (
        ("CAPTION", caption, config.caption_token_budget),
        ("VISIBLE_TEXT", ocr, config.ocr_token_budget),
        ("OBJECTS", objects, config.object_token_budget),
    ):
        normalized = _truncate_whitespace_tokens(text or "", budget)
        if normalized:
            sections.append(f"[{heading}]\n{normalized}")
    return "\n\n".join(sections) or None


__all__ = ["serialize_frame_context"]
