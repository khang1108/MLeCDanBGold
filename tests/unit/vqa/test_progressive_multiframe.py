from __future__ import annotations

from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    QueryLanguage,
    RetrievalSource,
    SceneCandidate,
)
from hcmai.pipelines.vqa.domain.models import (
    EvidenceBundle,
    ParsedVQAQuery,
    QuestionType,
)
from hcmai.pipelines.vqa.reasoning.answerer import answer_windows


class Image:
    def close(self):
        pass


class MultiFrameLLM:
    def __init__(self):
        self.frame_ids = None

    def answer_vqa_multi(self, *, frame_ids, **kwargs):
        self.frame_ids = frame_ids
        return {
            "answer": "hai",
            "selected_frame_id": frame_ids[-1],
            "answerable": True,
            "grounded": True,
            "confidence": 0.9,
        }


def test_answer_path_passes_bounded_frames_in_chronological_order():
    frames = tuple(
        FrameRecord(
            frame_id=frame_id,
            video_id="v1",
            frame_idx=index,
            timestamp_ms=timestamp,
            image_path=f"{frame_id}.jpg",
            width=10,
            height=10,
        )
        for index, (frame_id, timestamp) in enumerate((
            ("f1", 1_000), ("f2", 2_000), ("f3", 3_000),
        ))
    )
    evidence = tuple(
        FrameEvidence(frame=frame, unit_scores={"h0": 0.8}, score=0.8)
        for frame in frames
    )
    scene = SceneCandidate(
        scene_id="s1", video_id="v1", start_ms=1_000, end_ms=3_000,
        evidence=evidence, final_score=0.8,
    )
    parsed = ParsedVQAQuery(
        retrieval_query="cảnh có nhiều quả",
        question="Có bao nhiêu loại quả?",
        question_type=QuestionType.COUNT,
        required_modalities=(RetrievalSource.VISUAL,),
        answer_language=QueryLanguage.VIETNAMESE,
    )
    llm = MultiFrameLLM()
    answers, warnings = answer_windows(
        [EvidenceBundle(scene=scene, image_frames=frames)],
        parsed,
        llm,
        max_calls=1,
        image_loader=lambda _: Image(),
    )
    assert warnings == []
    assert llm.frame_ids == ["f1", "f2", "f3"]
    assert answers[0].evidence_frame_id == "f3"
