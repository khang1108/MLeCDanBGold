from __future__ import annotations

from dataclasses import replace

from hcmai.common.schemas import FrameRecord, RetrievalSource, VQARequest
from hcmai.vqa.answerer import answer_windows
from hcmai.vqa.evidence import build_evidence_bundle
from hcmai.vqa.localizer import SimilarityLocalizer
from hcmai.vqa.models import BranchCandidate, GroundedAnswerCandidate, VideoEvidenceCandidate
from hcmai.vqa.normalization import normalize_answer
from hcmai.vqa.parser import parse_vqa_query
from hcmai.vqa.ranking import rank_grounded_answers
from hcmai.vqa.submission import materialize_submissions
from hcmai.vqa.windows import build_windows


def frame(frame_id, video, index, timestamp):
    return FrameRecord(
        frame_id=frame_id, video_id=video, frame_idx=index, timestamp_ms=timestamp,
        image_path=f"/{frame_id}.jpg", width=10, height=10,
    )


class FakeData:
    def __init__(self, frames):
        self.frames = frames
        self.by_id = {item.frame_id: item for item in frames}

    def neighbors(self, frame_id, *, window_ms, include_self=False):
        target = self.by_id[frame_id]
        return [
            frame for frame in self.frames
            if frame.video_id == target.video_id
            and abs(frame.timestamp_ms - target.timestamp_ms) <= window_ms
            and (include_self or frame.frame_id != frame_id)
        ]

    def get_frame(self, frame_id):
        return self.by_id[frame_id]

    def get_evidence(self, frame_id, source):
        return "two people" if source == RetrievalSource.CAPTION else None


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def answer_vqa(self, question, image, evidence):
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value


def setup_localized():
    f1, f2 = frame("f1", "v1", 7, 1_000), frame("f2", "v2", 8, 1_000)
    data = FakeData([f1, f2])
    candidates = [
        BranchCandidate(f1, {"event": 0.9}, {RetrievalSource.VISUAL: 0.9}, {}, 0.9, ("event",)),
        BranchCandidate(f2, {"event": 0.8}, {RetrievalSource.VISUAL: 0.8}, {}, 0.8, ("event",)),
    ]
    videos = [VideoEvidenceCandidate(item.frame.video_id, (item,), item.score, 1, None, 1, 1, 0.0) for item in candidates]
    bundles = [build_evidence_bundle(window, data) for window in build_windows(videos, data, duration_ms=2_000)]
    parsed = parse_vqa_query(VQARequest(event_description="two people", question="How many people?"))
    return data, parsed, SimilarityLocalizer().localize(parsed, bundles, limit=2)


def test_multi_candidate_partial_failure_wrong_identity_and_deterministic_fallback():
    data, parsed, localized = setup_localized()
    llm = FakeLLM([
        {"answer": "two", "frame_id": "unknown", "confidence": "high"},
        TimeoutError("down"),
    ])
    answers, warnings = answer_windows(localized, parsed, llm, max_calls=2, image_loader=lambda _: object())
    assert answers == []
    assert warnings == ["provider_returned_unknown_frame_id", "vqa_provider_timeouterror"]


def test_answer_normalize_rank_and_materialize_canonical_identity():
    data, parsed, localized = setup_localized()
    frame_ids = [item.bundle.window.sampled_frames[0].frame_id for item in localized]
    llm = FakeLLM([
        {"answer": "Two!", "frame_id": frame_ids[0], "confidence": 0.8},
        {"answer": "2", "frame_id": frame_ids[1], "confidence": 0.6},
    ])
    answers, warnings = answer_windows(localized, parsed, llm, max_calls=2, image_loader=lambda _: object())
    assert not warnings
    assert {item.normalized_answer for item in answers} == {"2"}
    ranked = rank_grounded_answers(answers)
    rows = materialize_submissions(ranked, data, top_k=2)
    assert [row.rank for row in rows] == [1, 2]
    assert {(row.video_id, row.frame_idx) for row in rows} == {("v1", 7), ("v2", 8)}


def test_invalid_grounding_cannot_win_and_normalization_is_conservative():
    data, parsed, localized = setup_localized()
    window = localized[0].bundle.window
    base = GroundedAnswerCandidate(window, "f1", "two", "2", 0.1, 0.1, 0.1, 0.1, 0.1)
    invalid = replace(base, answer_confidence=1.0, grounded=False)
    assert rank_grounded_answers([invalid, base]) == [replace(
        base, consistency_score=1.0, joint_score=1.0,
        score_components={"video": 1.0, "frame": 1.0, "grounding": 1.0, "answer": 1.0, "consistency": 1.0},
    )]
    assert normalize_answer("  Xanh lá! ", parsed.question_type) == "xanh lá"
