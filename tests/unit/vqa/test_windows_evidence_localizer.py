from __future__ import annotations

from hcmai.common.schemas import FrameRecord, RetrievalSource, VQARequest
from hcmai.vqa.evidence import build_evidence_bundle
from hcmai.vqa.localizer import SimilarityLocalizer
from hcmai.vqa.models import BranchCandidate, VideoEvidenceCandidate
from hcmai.vqa.parser import parse_vqa_query
from hcmai.vqa.windows import build_windows, expand_neighbor_window


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


def test_merging_stays_inside_the_duration_and_keeps_every_anchor():
    frames = [frame(f"f{index}", index, index * 1_000) for index in range(31)]
    data = FakeData(frames)
    anchors = (candidate(frames[10], 0.9), candidate(frames[11], 0.8), candidate(frames[20], 0.7))

    windows = build_windows(
        [video_candidate(*anchors)], data, duration_ms=15_000, max_frames=4
    )

    assert [window.end_ms - window.start_ms for window in windows] == [15_000, 14_000]
    assert set(windows[0].frame_ids) >= {"f10", "f11"}
    assert "f20" in windows[1].frame_ids


def test_a_merge_that_would_outgrow_the_frame_budget_is_refused():
    frames = [frame(f"f{index}", index, index * 100) for index in range(30)]
    data = FakeData(frames)
    anchors = tuple(candidate(frames[10 + offset], 0.5 + 0.1 * offset) for offset in range(5))

    windows = build_windows(
        [video_candidate(*anchors)], data, duration_ms=15_000, max_frames=4
    )

    assert [len(window.source_frames) for window in windows] == [4, 1]
    for window in windows:
        assert {item.frame.frame_id for item in window.source_frames} <= set(window.frame_ids)
    assert "f14" in set(windows[0].frame_ids)


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
