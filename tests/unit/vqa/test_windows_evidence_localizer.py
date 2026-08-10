from __future__ import annotations

from hcmai.common.schemas import FrameRecord, RetrievalSource, VQARequest
from hcmai.pipelines.vqa.evidence import build_evidence_bundle
from hcmai.pipelines.vqa.localizer import SimilarityLocalizer
from hcmai.pipelines.vqa.models import BranchCandidate, VideoEvidenceCandidate
from hcmai.pipelines.vqa.parser import parse_vqa_query
from hcmai.pipelines.vqa.windows import build_windows, expand_neighbor_window


def frame(frame_id, index, timestamp):
    return FrameRecord(
        frame_id=frame_id, video_id="video", frame_idx=index, timestamp_ms=timestamp,
        image_path=f"/{frame_id}.jpg", width=10, height=10,
    )


class FakeData:
    def __init__(self, frames, evidence=None):
        self.frames = frames
        self.evidence = evidence or {}

    def neighbors(self, frame_id, *, window_ms, include_self=False):
        target = next(frame for frame in self.frames if frame.frame_id == frame_id)
        return [
            frame for frame in self.frames
            if frame.video_id == target.video_id
            and abs(frame.timestamp_ms - target.timestamp_ms) <= window_ms
            and (include_self or frame.frame_id != frame_id)
        ]

    def get_evidence(self, frame_id, source):
        return self.evidence.get((frame_id, source))


def candidate(value, score):
    return BranchCandidate(value, {"event": score}, {RetrievalSource.VISUAL: score}, {}, score, ("event",))


def video_candidate(*frames):
    return VideoEvidenceCandidate("video", tuple(frames), 1.0, 1, None, 1, 1, 0.0)


def test_windows_clamp_merge_and_expand_once():
    frames = [frame("f0", 0, 0), frame("f1", 1, 5_000), frame("f2", 2, 10_000), frame("f3", 3, 20_000)]
    data = FakeData(frames)
    windows = build_windows([video_candidate(candidate(frames[0], 1.0), candidate(frames[1], 0.9))], data, duration_ms=10_000)
    assert len(windows) == 1
    assert windows[0].start_ms == 0
    assert windows[0].end_ms == 10_000
    expanded = expand_neighbor_window(windows[0], data, expansion_ms=10_000)
    assert expanded is not None and expanded.end_ms == 20_000
    assert expand_neighbor_window(expanded, data, expansion_ms=10_000) is None


def test_evidence_deduplicates_text_and_localizer_is_deterministic():
    frames = [frame("f0", 0, 0), frame("f1", 1, 5_000)]
    data = FakeData(frames, {
        ("f0", RetrievalSource.CAPTION): "red car",
        ("f1", RetrievalSource.CAPTION): " red   car ",
        ("f1", RetrievalSource.OCR): "STOP",
    })
    windows = build_windows([video_candidate(candidate(frames[0], 1.0))], data, duration_ms=10_000)
    bundle = build_evidence_bundle(windows[0], data)
    assert [item.value for item in bundle.items] == ["red car", "STOP"]
    parsed = parse_vqa_query(VQARequest(event_description="a red car", question="What is written?"))
    localized = SimilarityLocalizer().localize(parsed, [bundle], limit=1)
    assert localized[0].reason_labels == ("retrieval_similarity", "lexical_overlap")


def test_missing_evidence_is_warning_not_negative_evidence():
    f0 = frame("f0", 0, 0)
    data = FakeData([f0])
    bundle = build_evidence_bundle(build_windows([video_candidate(candidate(f0, 1.0))], data)[0], data)
    assert bundle.items == ()
    assert bundle.warnings == ("text_evidence_unavailable",)
