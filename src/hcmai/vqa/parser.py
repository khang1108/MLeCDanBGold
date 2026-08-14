"""Deterministic Vietnamese/English competition VQA parser."""

from __future__ import annotations

import re
import unicodedata

from hcmai.common.schemas import QueryLanguage, RetrievalSource, VQARequest

from .models import ParsedVQAQuery, QuestionType


_RULES: tuple[tuple[QuestionType, str], ...] = (
    (QuestionType.COUNT, r"how many|bao nhiêu|\bmấy\b|số lượng"),
    (QuestionType.COLOR, r"what colou?r|which colou?r|màu gì|màu nào"),
    (
        QuestionType.TEXT,
        r"\bwritten\b|\b(text|sign|label|screen|board|banner|subtitle)s?\b"
        r"|chữ gì|ghi gì|viết gì|biển ghi|màn hình",
    ),
    (QuestionType.SPEECH, r"\b(says?|said|saying|tells?|told|speaks?|spoken)\b|nói gì|hỏi gì"),
    (QuestionType.TEMPORAL, r"\b(before|after|then|next)\b|\btrước\b|\bsau\b|tiếp theo"),
    (QuestionType.IDENTITY, r"\bwho\b|what is|which object|\bai\b|là gì|vật gì"),
)


def parse_vqa_query(request: VQARequest) -> ParsedVQAQuery:
    """Parse without inventing scene facts; event text remains retrieval truth."""

    event = _clean(request.event_description)
    question = _clean(request.question)
    if not event or not question:
        raise ValueError("event_description and question must be non-empty")
    folded = unicodedata.normalize("NFKC", question).casefold()
    question_type = next(
        (kind for kind, pattern in _RULES if re.search(pattern, folded)),
        QuestionType.GENERAL,
    )
    language = request.language_hint or _detect_language(folded)
    required = _modalities(question_type)
    clues = (question,) if question != event else ()
    return ParsedVQAQuery(
        retrieval_query=event,
        question=question,
        question_type=question_type,
        required_modalities=required,
        answer_language=language,
        clue_queries=clues,
    )


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _detect_language(value: str) -> QueryLanguage:
    if any(char in value for char in "ăâđêôơư") or re.search(
        r"\b(gì|nào|bao|người|trước|sau)\b", value
    ):
        return QueryLanguage.VIETNAMESE
    return QueryLanguage.ENGLISH


def _modalities(kind: QuestionType) -> tuple[RetrievalSource, ...]:
    preferred = {
        QuestionType.TEXT: (RetrievalSource.OCR, RetrievalSource.VISUAL),
        QuestionType.SPEECH: (RetrievalSource.ASR, RetrievalSource.VISUAL),
    }
    return preferred.get(kind, (RetrievalSource.VISUAL,))
