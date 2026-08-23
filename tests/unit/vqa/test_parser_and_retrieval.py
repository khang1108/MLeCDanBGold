from __future__ import annotations

from hcmai.common.schemas import (
    QueryLanguage,
    RetrievalSource,
    VQARequest,
)
from hcmai.pipelines.vqa.domain.models import QuestionType
from hcmai.pipelines.vqa.query.parser import parse_vqa_query


def test_parser_detects_vietnamese_ocr_and_preserves_raw_event():
    parsed = parse_vqa_query(VQARequest(
        event_description=" Người đi ngang biển báo. ", question="Biển ghi gì?",
    ))
    assert parsed.retrieval_query == "Người đi ngang biển báo."
    assert parsed.question_type == QuestionType.TEXT
    assert parsed.answer_language == QueryLanguage.VIETNAMESE
    assert parsed.required_modalities == (RetrievalSource.OCR, RetrievalSource.VISUAL)


def test_parser_detects_english_temporal_question():
    parsed = parse_vqa_query(VQARequest(
        event_description="A person enters a room", question="What happens after they sit?",
    ))
    assert parsed.question_type == QuestionType.TEMPORAL
    assert parsed.answer_language == QueryLanguage.ENGLISH
