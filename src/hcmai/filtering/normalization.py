"""Normalize Filter text consistently across offline and online paths.

This normalization intentionally removes Vietnamese diacritics for exact
substring filtering. It is separate from retrieval-query normalization,
which preserves text for embedding models.
"""

from __future__ import annotations

import unicodedata


def normalize_filter_text(value: str) -> str:
    """Return deterministic lowercase, accent-free, collapsed Filter text."""

    mapped = value.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", mapped)
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_marks.lower().split())

