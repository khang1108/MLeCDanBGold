from __future__ import annotations

from hcmai.common.schemas import FrameEvidence, FrameRecord
from hcmai.temporal.aligners.scene import cluster_video_evidence
from hcmai.temporal.state.evidence import ProgressiveEvidenceState
from hcmai.temporal.providers.sparse import candidate_video_scores, retain_top_evidence


def _item(frame_id: str, timestamp_ms: int, score: float) -> FrameEvidence:
    return FrameEvidence(
        frame=FrameRecord(
            frame_id=frame_id,
            video_id="v1",
            frame_idx=timestamp_ms,
            timestamp_ms=timestamp_ms,
            image_path=f"{frame_id}.jpg",
            width=10,
            height=10,
        ),
        unit_scores={"h0": score},
        score=score,
    )


def test_top_m_deduplicates_before_truncating():
    first = _item("f1", 100, 0.99)
    retained = retain_top_evidence(
        [first, first, _item("f2", 200, 0.90), _item("f3", 300, 0.89)],
        3,
    )
    assert [item.frame.frame_id for item in retained] == ["f1", "f2", "f3"]


def test_scene_clustering_caps_total_span_despite_chaining():
    evidence = [_item(f"f{index}", index * 2_500, 0.8) for index in range(8)]
    scenes = cluster_video_evidence(
        "v1",
        evidence,
        max_gap_ms=3_000,
        max_span_ms=10_000,
    )
    assert len(scenes) == 2
    assert all(scene.end_ms - scene.start_ms <= 10_000 for scene in scenes)


def test_candidate_score_prefers_multi_hint_coverage_and_tracks_unknown():
    state = ProgressiveEvidenceState()
    # Video A has one exceptional frame but fails the other three hints.
    state.mark_evaluated("h0", "a", (_item("a0", 0, 0.99),))
    for unit_id in ("h1", "h2", "h3"):
        state.mark_evaluated(unit_id, "a")
    # Video B consistently supports all four hints.
    for index, score in enumerate((0.82, 0.84, 0.79, 0.81)):
        item = _item(f"b{index}", index * 1_000, score).model_copy(
            update={
                "frame": _item(f"b{index}", index * 1_000, score).frame.model_copy(
                    update={"video_id": "b"}
                ),
                "unit_scores": {f"h{index}": score},
            }
        )
        state.mark_evaluated(f"h{index}", "b", (item,))
    scores = candidate_video_scores(
        state,
        unit_ids=["h0", "h1", "h2", "h3"],
        allowed_video_ids={"a", "b"},
        semantic_weight=0.45,
        match_weight=0.25,
        evaluation_weight=0.30,
    )
    assert scores["b"] > scores["a"]

    # A newly rescued video with one match remains promising, but its three
    # UNKNOWN units do not make it appear perfectly evaluated.
    rescued = _item("r3", 3_000, 0.9).model_copy(
        update={
            "frame": _item("r3", 3_000, 0.9).frame.model_copy(
                update={"video_id": "rescued"}
            ),
            "unit_scores": {"h3": 0.9},
        }
    )
    state.mark_evaluated("h3", "rescued", (rescued,))
    scores = candidate_video_scores(
        state,
        unit_ids=["h0", "h1", "h2", "h3"],
        allowed_video_ids={"b", "rescued"},
        semantic_weight=0.45,
        match_weight=0.25,
        evaluation_weight=0.30,
    )
    assert scores["b"] > scores["rescued"]
