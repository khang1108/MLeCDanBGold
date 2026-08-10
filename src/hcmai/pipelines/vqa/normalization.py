"""Conservative deterministic short-answer normalization."""

from __future__ import annotations

import re
import unicodedata

from .models import QuestionType


_COUNTS = {
    "zero": "0", "không": "0", "one": "1", "một": "1", "mot": "1",
    "two": "2", "hai": "2", "three": "3", "ba": "3", "four": "4", "bốn": "4",
    "five": "5", "năm": "5", "six": "6", "sáu": "6", "seven": "7", "bảy": "7",
    "eight": "8", "tám": "8", "nine": "9", "chín": "9", "ten": "10", "mười": "10",
}
_COLORS = {
    "red": "red", "đỏ": "đỏ", "blue": "blue", "xanh dương": "xanh dương",
    "green": "green", "xanh lá": "xanh lá", "yellow": "yellow", "vàng": "vàng",
    "black": "black", "đen": "đen", "white": "white", "trắng": "trắng",
}


def normalize_answer(answer: str, question_type: QuestionType) -> str:
    value = unicodedata.normalize("NFC", answer).strip().casefold()
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    if question_type == QuestionType.COUNT:
        return _COUNTS.get(value, value)
    if question_type == QuestionType.COLOR:
        return _COLORS.get(value, value)
    return value
