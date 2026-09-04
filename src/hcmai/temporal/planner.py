"""Deterministically turn a KIS query into ordered timeline moments.

This module owns only backend query segmentation for the baseline temporal
search flow. It does not validate TRAKE requests, infer event meaning with an
LLM, retrieve frames, or perform temporal alignment.
"""

from __future__ import annotations

import re

from collections.abc import Sequence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_QUESTION = re.compile(
    r"^(hỏi|hãy cho biết)\b|\?\s*$|\b("
    r"bao nhiêu|gì|số mấy|mấy giờ"
    r"|(số|vùng|đường|nơi|chỗ|loại|màu|tên|hình|người|con|cái|thứ) nào"
    r"|ở đâu|khi nào|tại sao|vì sao|thế nào"
    r")\b",
    re.IGNORECASE,
)
_ATTRIBUTE = re.compile(
    r"^(-|•|có thể thấy|có \d|có thông tin|trong nhóm|trong đó|khung hình"
    r"|xuất hiện trong|góc máy|ảnh nền|bảng này|từ bảng|kế bên|xung quanh"
    r"|bên (cạnh|dưới|phải|trái|trong)|phía (sau|trước|trên|dưới)"
    r"|trên (băng|màn|slide|bảng|đó))",
    re.IGNORECASE,
)
_EARLIER = re.compile(r"^trước đó\b", re.IGNORECASE)
_LATER = re.compile(r"\b(tiếp theo|sau đó|kế tiếp|ngay sau|cuối cùng là)\b", re.IGNORECASE)


def split_query_events(query: str) -> tuple[str, ...]:
    """Split one query into deterministic ordered event text.

    Nonempty lines take precedence over sentence boundaries, and a single
    remaining query becomes one event. Whitespace is normalized without
    semantic rewriting. The return value is intentionally a plain tuple so the
    baseline has no alignment DTO.
    """

    lines = _normalize_parts(query.splitlines(), strip_terminal_punctuation=False)
    if len(lines) >= 2:
        parts = lines
    else:
        sentences = _normalize_parts(
            _SENTENCE_BOUNDARY.split(query),
            strip_terminal_punctuation=True,
        )
        parts = (
            sentences
            if len(sentences) >= 2
            else _normalize_parts(
                [query],
                strip_terminal_punctuation=False,
            )
        )

    if not parts:
        raise ValueError("alignment query must contain at least one event")

    return tuple(parts)


def plan_query_events(query: str) -> tuple[str, ...]:
    """Split one query into timeline moments rather than into sentences.

    Sentences are segmented by :func:`split_query_events`, then folded: a
    trailing question is the reviewer's job and never a moment, a sentence
    opening with an attribute cue describes the previous moment, an enumeration
    under a colon belongs to the sentence that introduced it, and ``Trước đó``
    marks a moment that happened before the one just described.
    """

    parts = list(split_query_events(query))
    if len(parts) > 1 and _QUESTION.search(parts[-1]):
        parts.pop()

    planned: list[str] = []
    enumerating = False
    for part in parts:
        attribute = enumerating or (_ATTRIBUTE.match(part) and not _LATER.search(part))
        if planned and attribute:
            planned[-1] = f"{planned[-1]} {part}"
        elif planned and _EARLIER.match(part):
            planned.insert(-1, part)
        else:
            planned.append(part)
        enumerating = enumerating or part.rstrip().endswith(":")
    return tuple(planned)


def _normalize_parts(
    values: Sequence[str],
    *,
    strip_terminal_punctuation: bool,
) -> list[str]:
    """Normalize whitespace and remove only sentence delimiters when splitting."""

    parts = [" ".join(value.split()) for value in values]
    if strip_terminal_punctuation:
        parts = [part.rstrip(".!?").rstrip() for part in parts]
    return [part for part in parts if part]
