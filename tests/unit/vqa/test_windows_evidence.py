from __future__ import annotations

from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    RetrievalSource,
    SceneCandidate,
)
from hcmai.pipelines.vqa.domain.models import EvidenceBundle
from hcmai.pipelines.vqa.reasoning.evidence import build_evidence_bundle
from hcmai.pipelines.vqa.reasoning.windows import expand_neighbor_window


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
    return FrameEvidence(
        frame=value,
        unit_scores={"event": score},
        source_scores={RetrievalSource.VISUAL: score},
        score=score,
        provenance=("event",),
    )


def scene_bundle(frames, evidence):
    scene = SceneCandidate(
        scene_id="video:0-10000",
        video_id="video",
        start_ms=0,
        end_ms=10_000,
        evidence=tuple(evidence),
        final_score=1.0,
    )
    return EvidenceBundle(scene=scene, image_frames=tuple(frames))


def test_neighbor_expansion_clamps_and_expands_once():
    frames = [
        frame("f0", 0, 0),
        frame("f1", 1, 5_000),
        frame("f2", 2, 10_000),
        frame("f3", 3, 20_000),
    ]
    data = FakeData(frames)
    initial = scene_bundle(
        frames[:3],
        [candidate(frames[0], 1.0), candidate(frames[1], 0.9)],
    )
    expanded = expand_neighbor_window(initial, data, expansion_ms=10_000)
    assert expanded is not None and expanded.scene.end_ms == 20_000
    assert expand_neighbor_window(expanded, data, expansion_ms=10_000) is None


def test_evidence_deduplicates_text():
    frames = [frame("f0", 0, 0), frame("f1", 1, 5_000)]
    data = FakeData(frames, {
        ("f0", RetrievalSource.CAPTION): "red car",
        ("f1", RetrievalSource.CAPTION): " red   car ",
        ("f1", RetrievalSource.OCR): "STOP",
    })
    evidence = build_evidence_bundle(
        scene_bundle(frames, [candidate(frames[0], 1.0)]),
        data,
    )
    assert [item.value for item in evidence.items] == ["red car", "STOP"]


def test_missing_evidence_is_warning_not_negative_evidence():
    f0 = frame("f0", 0, 0)
    data = FakeData([f0])
    evidence = build_evidence_bundle(
        scene_bundle([f0], [candidate(f0, 1.0)]),
        data,
    )
    assert evidence.items == ()
    assert evidence.warnings == ("text_evidence_unavailable",)
