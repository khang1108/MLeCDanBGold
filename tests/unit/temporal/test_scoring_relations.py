from __future__ import annotations

from hcmai.common.config import ProgressiveSearchConfig
from hcmai.common.schemas import (
    FrameEvidence,
    FrameRecord,
    QueryUnit,
    SceneCandidate,
    TemporalConstraint,
    TemporalRelation,
)
from hcmai.temporal.evidence import ProgressiveEvidenceState
from hcmai.temporal.relations import parse_temporal_constraints
from hcmai.temporal.scoring import rank_scenes, score_scene


def _item(frame_id: str, unit_id: str, timestamp: int, score: float):
    return FrameEvidence(
        frame=FrameRecord(
            frame_id=frame_id, video_id="v1", frame_idx=timestamp // 1000,
            timestamp_ms=timestamp, image_path=f"{frame_id}.jpg", width=10, height=10,
        ),
        unit_scores={unit_id: score},
        score=score,
    )


def test_scene_components_are_named_bounded_and_config_driven():
    units = [QueryUnit(unit_id="h0", text="A", order=0), QueryUnit(unit_id="h1", text="B", order=1)]
    evidence = (_item("f1", "h0", 1_000, 0.6), _item("f2", "h1", 2_000, 0.6))
    state = ProgressiveEvidenceState()
    state.mark_evaluated("h0", "v1", (evidence[0],))
    state.mark_evaluated("h1", "v1", (evidence[1],))
    scene = SceneCandidate(scene_id="s", video_id="v1", start_ms=1_000, end_ms=2_000, evidence=evidence)
    first = score_scene(scene, units, state, [], ProgressiveSearchConfig(), coherence_window_ms=10_000)
    second = score_scene(
        scene, units, state, [],
        ProgressiveSearchConfig(
            scene_semantic_weight=1, scene_coverage_weight=0,
            scene_temporal_weight=0, scene_relation_weight=0,
        ),
        coherence_window_ms=10_000,
    )
    assert first.semantic_score == second.semantic_score == 0.6
    assert first.coverage_score == second.coverage_score == 1.0
    assert first.relation_score is None
    assert first.final_score != second.final_score
    assert all(0 <= value <= 1 for value in (
        first.semantic_score, first.coverage_score, first.temporal_score,
        first.final_score,
    ))


def test_unknown_is_masked_and_ties_are_deterministic():
    units = [QueryUnit(unit_id="h0", text="A", order=0), QueryUnit(unit_id="h1", text="B", order=1)]
    item = _item("f1", "h0", 1_000, 0.5)
    state = ProgressiveEvidenceState()
    state.mark_evaluated("h0", "v1", (item,))
    scene = SceneCandidate(scene_id="b", video_id="v1", start_ms=1_000, end_ms=1_000, evidence=(item,))
    scored = score_scene(scene, units, state, [], ProgressiveSearchConfig(), coherence_window_ms=10_000)
    assert scored.coverage_score == 1.0
    assert scored.evaluation_coverage_score == 0.5
    tied = [scored.model_copy(update={"scene_id": "b"}), scored.model_copy(update={"scene_id": "a"})]
    assert [item.scene_id for item in rank_scenes(tied)] == ["a", "b"]


def test_relation_parser_requires_explicit_language():
    ordinary = [QueryUnit(unit_id="h0", text="nồi xanh", order=0), QueryUnit(unit_id="h1", text="nồi nâu", order=1)]
    assert parse_temporal_constraints(ordinary) == []
    supported = [
        ("sau đó B", "before"),
        ("cuối cùng X", "at_end"),
        ("đồng thời B", "overlap"),
    ]
    for text, relation in supported:
        constraints = parse_temporal_constraints([
            QueryUnit(unit_id="h0", text="A", order=0),
            QueryUnit(unit_id="h1", text=text, order=1),
        ])
        assert constraints[0].relation.value == relation
    for ambiguous in ("B trước đó là A", "sau khi A, B", "ngay trước B"):
        assert parse_temporal_constraints([
            QueryUnit(unit_id="h0", text="A", order=0),
            QueryUnit(unit_id="h1", text=ambiguous, order=1),
        ]) == []


def test_relation_uses_any_valid_evidence_pair_and_unknown_stays_unknown():
    constraint = TemporalConstraint(
        relation=TemporalRelation.BEFORE,
        subject_unit_id="h0",
        object_unit_id="h1",
        reason="test_before",
    )
    evidence = (
        _item("h0-10", "h0", 10_000, 0.8),
        _item("h0-100", "h0", 100_000, 0.8),
        _item("h1-5", "h1", 5_000, 0.8),
        _item("h1-50", "h1", 50_000, 0.8),
    )
    from hcmai.temporal.relations import relation_satisfaction

    score, _ = relation_satisfaction([constraint], evidence, near_ms=1_000)
    assert score == 1.0
    score, labels = relation_satisfaction(
        [constraint],
        (evidence[0],),
        near_ms=1_000,
    )
    assert score is None
    assert labels == ("relation_unknown:test_before",)
