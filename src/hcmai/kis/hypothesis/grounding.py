"""Source grounding and language classification for KIS Query Hypotheses.

This module provides deterministic source alignment for LLM-proposed event
fragments against the canonical original query, and server-owned query language
detection.
"""

from __future__ import annotations

from collections.abc import Sequence
import unicodedata

from hcmai.kis.models import SourceProvenance

_VI_MARKERS = set("ăâđêôơưĂÂĐÊÔƠƯ")

# Lowercase ASCII Vietnamese function words that rarely appear in English text.
# Conservative: threshold of 2 distinct hits to classify as 'vi', avoiding
# accidental misclassification of short English snippets.
_VI_ASCII_WORDS: frozenset[str] = frozenset({
    "nguoi", "cua", "trong", "ngoai", "voi", "sau", "truoc", "tren", "duoi",
    "khi", "nhieu", "cung", "hoac", "nhung", "cac", "thay", "nhin",
    "buoc", "ngoi", "chay", "tat", "lay", "dem", "dat", "nhat", "nay",
    "vao", "den", "thi", "roi", "chiec",
})

_VI_ASCII_THRESHOLD: int = 2


def infer_query_language(query: str) -> str:
    """Infer server-owned query language: 'vi', 'en', or 'mixed'.

    Classification is deterministic and bounded to 'vi'/'en'/'mixed'.

    Vietnamese detection uses two independent signals:
    1. Unicode diacritics (NFC/NFD) — high precision.
    2. ASCII Vietnamese function-word heuristic — catches unaccented input such as
       'nguoi dan ong buoc vao phong', which would otherwise be classified as 'en'
       and cause _dense_projection() to skip EN→VI translation.
    """
    normalized = unicodedata.normalize("NFC", query)
    has_vi = any(
        ch in _VI_MARKERS or ("\u0300" <= ch <= "\u036f")
        for ch in unicodedata.normalize("NFD", normalized)
    )
    has_ascii_word = any(
        part.isascii() and part.isalpha() for part in normalized.split()
    )
    if has_vi and has_ascii_word:
        return "mixed"
    if has_vi:
        return "vi"

    # ASCII-only path: check for Vietnamese function words.
    # Require at least _VI_ASCII_THRESHOLD distinct hits to avoid false positives
    # on short English phrases that happen to contain one ambiguous token.
    lowered_tokens = {
        part.lower()
        for part in normalized.split()
        if part.isascii() and part.isalpha()
    }
    vi_hits = lowered_tokens & _VI_ASCII_WORDS
    if len(vi_hits) >= _VI_ASCII_THRESHOLD:
        return "vi"
    return "en"



def align_source_fragments(
    query: str, fragments: Sequence[str]
) -> tuple[SourceProvenance, ...]:
    """Align verbatim event fragments sequentially left-to-right against the query.

    Args:
        query: Canonical original query text.
        fragments: Ordered sequence of verbatim fragments proposed by LLM.

    Returns:
        Tuple of validated SourceProvenance spans.

    Raises:
        ValueError: If any fragment cannot be verified left-to-right in query.
    """
    cursor = 0
    spans: list[SourceProvenance] = []
    for raw in fragments:
        fragment = " ".join(raw.split())
        start = query.find(fragment, cursor)
        if start < 0:
            raise ValueError(f"fragment is not grounded in source query: {fragment!r}")
        end = start + len(fragment)
        spans.append(
            SourceProvenance(source_text=fragment, start_char=start, end_char=end)
        )
        cursor = end
    return tuple(spans)
