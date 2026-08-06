"""Safe-by-default helpers for values that may contain user content."""

from __future__ import annotations

from typing import Any

REDACTED = "[REDACTED]"


def safe_content(value: Any, *, debug: bool = False, limit: int = 120) -> str:
    """Hide prompts, answers, and image payloads unless debug is explicit."""

    if not debug:
        return REDACTED
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}..."
