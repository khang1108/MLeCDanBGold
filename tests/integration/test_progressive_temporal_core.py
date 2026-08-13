from __future__ import annotations

import pytest

from hcmai.common.config import ProgressiveSearchConfig, SearchConfig
from hcmai.common.schemas import (
    FrameRecord,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalSource,
    SearchFilters,
    StageStatus,
    StageTrace,
    RetrievalTrace,
)
from hcmai.temporal import (
    ProgressiveStateConflictError,
    TemporalEvidenceCore,
)
from hcmai.temporal.evidence import EvaluationState
from hcmai.data.pipeline import DataService
from hcmai.orchestration.pipeline import SearchService
from hcmai.retrieval.retriever.pipeline import RetrievalService
from hcmai.common.schemas import TaskType


class Data:
    def __init__(self):
        self.frames = {
            "d1": _frame("d1", "distractor", 10_000, 10),
            "t1": _frame("t1", "target", 42_000, 42),
            "t2": _frame("t2", "target", 47_000, 47),
        }

    def get_frame(self, frame_id):
        return self.frames[frame_id]


class Retrieval:
    def __init__(self, *, fail_on: str | None = None):
        self.fail_on = fail_on
        self.calls = []

    def search(self, query, top_k=100, filters=None, query_type=None):
        self.calls.append((query, tuple(filters.video_ids) if filters else ()))
        if query == self.fail_on:
            raise RuntimeError("retrieval failed")
        videos = set(filters.video_ids) if filters and filters.video_ids else None
        if query == "H1 vague":
            ids = ["t1"] if videos == {"target"} else ["d1"]
        elif query == "H2 distinctive":
            ids = ["t2"] if videos in (None, {"target"}) else []
        else:
            ids = []
        return RetrievalResult(candidates=[
            RetrievalCandidate(
                frame_id=frame_id,
                source_scores={RetrievalSource.VISUAL: 0.8 if frame_id == "d1" else 0.9},
                source_ranks={RetrievalSource.VISUAL: rank},
                final_score=0.8 if frame_id == "d1" else 0.9,
            )
            for rank, frame_id in enumerate(ids[:top_k], start=1)
        ])


def _frame(frame_id, video_id, timestamp_ms, frame_idx):
    return FrameRecord(
        frame_id=frame_id, video_id=video_id, frame_idx=frame_idx,
        timestamp_ms=timestamp_ms, image_path=f"{frame_id}.jpg", width=10, height=10,
    )


def _config():
    return SearchConfig(
        temporal_window_ms=10_000,
        progressive=ProgressiveSearchConfig(
            candidate_pool_size=5,
            global_quota=5,
            local_quota=5,
            top_m_evidence=2,
            backfill_max_videos=5,
            backfill_max_units_per_video=5,
            scene_top_b_per_video=2,
            scene_top_p_global=5,
        ),
    )


def test_progressive_rescue_backfills_unknown_and_preserves_canonical_identity():
    retrieval = Retrieval()
    core = TemporalEvidenceCore(Data(), retrieval, _config())
    first = core.localize("H1 vague", search_id=None, filters=None)
    second = core.localize(
        "H1 vague\nH2 distinctive", search_id=first.search_id, filters=None
    )
    state = core.store.get(first.search_id)

    assert [unit.text for unit in state.query_units] == ["H1 vague", "H2 distinctive"]
    assert state.evidence.evaluation_state("h0", "target") is EvaluationState.MATCHED
    assert state.evidence.get_evidence("h0", "target")[0].frame is not None
    target_scene = next(scene for scene in second.scenes if scene.video_id == "target")
    assert {item.frame.frame_id for item in target_scene.evidence} == {"t1", "t2"}
    assert second.version == 2
    assert second.diagnostics["top_m_evidence"] == 2


def test_failed_update_leaves_committed_state_unchanged_then_retry_succeeds():
    retrieval = Retrieval(fail_on="H2 distinctive")
    core = TemporalEvidenceCore(Data(), retrieval, _config())
    first = core.localize("H1 vague", search_id=None, filters=None)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        core.localize("H1 vague H2 distinctive", search_id=first.search_id, filters=None)
    state = core.store.get(first.search_id)
    assert state.version == 1
    assert state.last_snapshot == "H1 vague"
    assert [unit.unit_id for unit in state.query_units] == ["h0"]

    retrieval.fail_on = None
    retry = core.localize(
        "H1 vague H2 distinctive", search_id=first.search_id, filters=None
    )
    assert retry.version == 2


def test_failed_first_request_creates_no_state():
    core = TemporalEvidenceCore(Data(), Retrieval(fail_on="H1 vague"), _config())
    with pytest.raises(RuntimeError, match="retrieval failed"):
        core.localize("H1 vague", search_id=None, filters=None)
    assert len(core.store) == 0


def test_noop_does_not_increment_and_rewrite_conflicts():
    core = TemporalEvidenceCore(Data(), Retrieval(), _config())
    first = core.localize("H1 vague", search_id=None, filters=None)
    noop = core.localize(" H1   vague. ", search_id=first.search_id, filters=None)
    assert noop.version == first.version
    with pytest.raises(ProgressiveStateConflictError, match="not a safe cumulative"):
        core.localize("rewritten clue", search_id=first.search_id, filters=None)


def test_progressive_hint_budget_is_enforced():
    config = _config().model_copy(update={
        "progressive": _config().progressive.model_copy(update={"progressive_max_hints": 1}),
    })
    core = TemporalEvidenceCore(Data(), Retrieval(), config)
    first = core.localize("H1 vague", search_id=None, filters=None)
    with pytest.raises(ProgressiveStateConflictError, match="hint limit"):
        core.localize("H1 vague H2 distinctive", search_id=first.search_id, filters=None)


def test_identical_kis_vqa_hint_history_produces_identical_pre_answer_scenes():
    first_core = TemporalEvidenceCore(Data(), Retrieval(), _config())
    second_core = TemporalEvidenceCore(Data(), Retrieval(), _config())
    histories = ["H1 vague", "H1 vague H2 distinctive"]
    ids = [None, None]
    outputs = []
    for index, core in enumerate((first_core, second_core)):
        result = core.localize(histories[0], search_id=None, filters=None)
        ids[index] = result.search_id
        outputs.append(core.localize(histories[1], search_id=result.search_id, filters=None))
    assert [scene.model_dump() for scene in outputs[0].scenes] == [
        scene.model_dump() for scene in outputs[1].scenes
    ]


def test_all_task_heads_receive_one_shared_temporal_facade():
    service = SearchService(DataService(), RetrievalService(Retrieval()), config=_config())
    kis = service.pipeline_registry.get(TaskType.KIS)
    vqa = service.pipeline_registry.get(TaskType.VQA)
    trake = service.pipeline_registry.get(TaskType.TRAKE)
    assert kis.temporal_core is vqa.temporal_core
    assert kis.temporal_core is not None
    assert kis.temporal_core is trake.temporal_core


def test_multi_video_top_k_absence_remains_unknown():
    class AmbiguousRetrieval(Retrieval):
        def search(self, query, top_k=100, filters=None, query_type=None):
            videos = set(filters.video_ids) if filters and filters.video_ids else None
            self.calls.append((query, tuple(filters.video_ids) if filters else ()))
            if query == "H1 vague":
                ids = ["d1", "t1"]
            elif query == "H2 crowded" and videos == {"distractor", "target"}:
                ids = ["d1"]
            elif query == "H2 crowded" and videos == {"target"}:
                ids = ["t2"]
            else:
                ids = []
            return RetrievalResult(candidates=[
                RetrievalCandidate(frame_id=frame_id, final_score=0.9)
                for frame_id in ids[:top_k]
            ])

    core = TemporalEvidenceCore(Data(), AmbiguousRetrieval(), _config())
    first = core.localize("H1 vague", search_id=None, filters=None)
    core.localize("H1 vague H2 crowded", search_id=first.search_id, filters=None)
    state = core.store.get(first.search_id)
    # The pooled local result omitted target, then a single-video backfill
    # evaluated it explicitly and recovered its evidence.
    assert state.evidence.evaluation_state("h1", "target") is EvaluationState.MATCHED


def test_progressive_session_rejects_task_filter_and_question_changes():
    core = TemporalEvidenceCore(Data(), Retrieval(), _config())
    filters = SearchFilters(video_ids=["target"])
    first = core.localize(
        "H1 vague",
        search_id=None,
        filters=filters,
        task_type=TaskType.VQA,
        session_fingerprint="question-a",
    )
    with pytest.raises(ProgressiveStateConflictError, match="belongs to vqa"):
        core.localize(
            "H1 vague H2 distinctive",
            search_id=first.search_id,
            filters=filters,
            task_type=TaskType.KIS,
            session_fingerprint="question-a",
        )
    with pytest.raises(ProgressiveStateConflictError, match="filters cannot change"):
        core.localize(
            "H1 vague H2 distinctive",
            search_id=first.search_id,
            filters=None,
            task_type=TaskType.VQA,
            session_fingerprint="question-a",
        )
    with pytest.raises(ProgressiveStateConflictError, match="context changed"):
        core.localize(
            "H1 vague H2 distinctive",
            search_id=first.search_id,
            filters=filters,
            task_type=TaskType.VQA,
            session_fingerprint="question-b",
        )


def test_progressive_result_preserves_retrieval_stage_traces():
    class TracedRetrieval(Retrieval):
        def search(self, query, top_k=100, filters=None, query_type=None):
            result = super().search(query, top_k, filters, query_type)
            stage = StageTrace(
                stage="search",
                started_at=1.0,
                ended_at=1.01,
                duration_ms=10.0,
                status=StageStatus.SUCCESS,
            )
            return result.model_copy(update={
                "trace": RetrievalTrace(stages={"search": stage}),
            })

    core = TemporalEvidenceCore(Data(), TracedRetrieval(), _config())
    result = core.localize("H1 vague", search_id=None, filters=None)
    assert "global.search" in result.trace.stages
