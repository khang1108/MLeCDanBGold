from __future__ import annotations

import pytest

from hcmai.common.schemas import (
    FrameRecord,
    QueryLanguage,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    VQARequest,
)
from hcmai.vqa.candidates import retrieve_candidates
from hcmai.vqa.models import QuestionType
from hcmai.vqa.parser import parse_vqa_query
from hcmai.vqa.video_aggregation import aggregate_videos


def frame(frame_id: str, video: str, index: int, timestamp: int) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id, video_id=video, frame_idx=index,
        timestamp_ms=timestamp, image_path=f"/{frame_id}.jpg", width=10, height=10,
    )


class FakeData:
    def __init__(self, frames):
        self.frames = {item.frame_id: item for item in frames}

    def get_frame(self, frame_id):
        return self.frames[frame_id]


class FakeRetrieval:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_batch(self, queries, top_k, filters, query_type):
        self.calls.append((queries, top_k, query_type))
        return self.results


def test_parser_detects_vietnamese_ocr_and_preserves_raw_event():
    parsed = parse_vqa_query(VQARequest(
        event_description=" Người đi ngang biển báo. ", question="Biển ghi gì?",
    ))
    assert parsed.retrieval_query == "Người đi ngang biển báo."
    assert parsed.question_type == QuestionType.TEXT
    assert parsed.answer_language == QueryLanguage.VIETNAMESE
    assert parsed.required_modalities == (RetrievalSource.OCR, RetrievalSource.VISUAL)


def test_a_spoken_question_is_asr_and_only_a_named_surface_is_ocr():
    expected = {
        "What does he say?": QuestionType.SPEECH,
        "What does the man say to her?": QuestionType.SPEECH,
        "What does the sign say?": QuestionType.TEXT,
        "What does that screen say?": QuestionType.TEXT,
        "What is written on the board?": QuestionType.TEXT,
        "Which object is already broken?": QuestionType.IDENTITY,
        "What is he spreading?": QuestionType.IDENTITY,
        "Người đó nói gì?": QuestionType.SPEECH,
        "Biển ghi gì?": QuestionType.TEXT,
    }

    parsed = {
        question: parse_vqa_query(
            VQARequest(event_description="a street scene", question=question)
        )
        for question in expected
    }

    assert {question: item.question_type for question, item in parsed.items()} == expected
    assert parsed["What does he say?"].required_modalities == (
        RetrievalSource.ASR, RetrievalSource.VISUAL,
    )
    assert parsed["What does the sign say?"].required_modalities == (
        RetrievalSource.OCR, RetrievalSource.VISUAL,
    )


def test_parser_detects_english_temporal_question():
    parsed = parse_vqa_query(VQARequest(
        event_description="A person enters a room", question="What happens after they sit?",
    ))
    assert parsed.question_type == QuestionType.TEMPORAL
    assert parsed.answer_language == QueryLanguage.ENGLISH


def test_retrieval_merges_duplicate_frame_identity_and_aggregates_consistency():
    f1, f2, f3 = frame("f1", "v1", 1, 1_000), frame("f2", "v1", 2, 20_000), frame("f3", "v2", 1, 1_000)
    event = RetrievalResult(candidates=[
        RetrievalCandidate(frame_id="f1", final_score=0.9, source_scores={RetrievalSource.VISUAL: 0.9}),
        RetrievalCandidate(frame_id="f3", final_score=1.0, source_scores={RetrievalSource.VISUAL: 1.0}),
    ])
    question = RetrievalResult(candidates=[
        RetrievalCandidate(frame_id="f1", final_score=0.8, source_scores={RetrievalSource.OCR: 0.8}),
        RetrievalCandidate(frame_id="f2", final_score=0.85, source_scores={RetrievalSource.OCR: 0.85}),
    ], warnings=["asr_unavailable"])
    retrieval = FakeRetrieval([event, question])
    parsed = parse_vqa_query(VQARequest(event_description="A sign appears", question="What is written?"))
    merged, warnings = retrieve_candidates(retrieval, FakeData([f1, f2, f3]), parsed)
    assert len(merged) == 3
    assert next(item for item in merged if item.frame.frame_id == "f1").provenance == ("event", "question")
    assert warnings == ["asr_unavailable"]
    videos = aggregate_videos(merged, top_videos=2)
    assert {item.video_id for item in videos} == {"v1", "v2"}
    assert next(item for item in videos if item.video_id == "v1").modality_count == 2


def test_the_question_modality_outranks_a_stronger_visual_only_frame():
    ocr, *visual = (frame(f"f{index}", "v1", index, index * 1_000) for index in range(4))
    result = RetrievalResult(candidates=[
        RetrievalCandidate(frame_id="f0", fusion_score=0.014, source_scores={RetrievalSource.OCR: 0.95}),
        *(
            RetrievalCandidate(frame_id=item.frame_id, fusion_score=0.016, source_scores={RetrievalSource.VISUAL: 0.3})
            for item in visual
        ),
    ])
    parsed = parse_vqa_query(VQARequest(event_description="A sign appears", question="What is written?"))

    merged, _ = retrieve_candidates(
        FakeRetrieval([result]), FakeData([ocr, *visual]), parsed, event_only=True
    )

    assert [item.frame.frame_id for item in merged] == ["f0", "f1", "f2", "f3"]
    assert merged[0].source_scores == {RetrievalSource.OCR: 0.95}
    general = parse_vqa_query(VQARequest(event_description="A sign appears", question="Describe it"))
    plain, _ = retrieve_candidates(
        FakeRetrieval([result]), FakeData([ocr, *visual]), general, event_only=True
    )
    assert [item.frame.frame_id for item in plain] == ["f1", "f2", "f3", "f0"]


def test_retrieval_flags_are_mutually_exclusive():
    parsed = parse_vqa_query(VQARequest(event_description="event", question="question"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        retrieve_candidates(FakeRetrieval([]), FakeData([]), parsed, event_only=True, question_only=True)
