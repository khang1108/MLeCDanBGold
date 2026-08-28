"""Tests for local archive/batch/video resume state persistence.

Covers atomic create/resume, changed-identity rejection, adjacent/gap work
windows, cleaned-overlap idempotent replay, ordered stage transitions,
skipped/reversed transition rejection, deterministic batch IDs, the
eight-video ceiling, bounded failure history, and the ephemeral-cleanup guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hcmai.data.custom_pipeline.config import ArchiveWorkWindow
from hcmai.data.custom_pipeline.contracts import RunIdentity
from hcmai.data.custom_pipeline.state import (
    ArchiveStage,
    BatchStage,
    PipelineStateStore,
    VideoStage,
    compute_batch_id,
)


def _identity(version: str = "dataset_v1") -> RunIdentity:
    return RunIdentity(
        version=version,
        source="custom_raw_video_1fps",
        frame_store_id="dataset_1",
        media_info_digest="a" * 64,
        archive_plan_digest="b" * 64,
        artifact_config_fingerprint="c" * 64,
        model_revisions={"caption": "rev1"},
        asr_lineage_digest="d" * 64,
    )


# ---------------------------------------------------------------------------
# Run identity + work window
# ---------------------------------------------------------------------------


def test_create_run_persists_identity_and_first_window(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    record = store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    assert record["identity"]["version"] == "dataset_v1"
    assert record["work_windows"] == [
        {"offset": 0, "limit": 1, "accepted_at": record["work_windows"][0]["accepted_at"]}
    ]
    # File is atomically persisted and reloadable.
    reloaded = PipelineStateStore(tmp_path)
    again = reloaded.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    assert len(again["work_windows"]) == 2  # replay is accepted as a new attempt record


def test_changed_identity_is_rejected(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    with pytest.raises(ValueError, match="identity changed"):
        store.create_or_resume_run(_identity(version="dataset_v2"), ArchiveWorkWindow(offset=0, limit=1))


def test_adjacent_work_window_requires_prior_archive_cleaned(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    store.ensure_archive("L01", position=0)

    with pytest.raises(ValueError, match="not cleaned"):
        store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=1, limit=1))

    for stage in (
        ArchiveStage.DOWNLOADING,
        ArchiveStage.DOWNLOADED,
        ArchiveStage.EXTRACTED,
        ArchiveStage.PROCESSING,
        ArchiveStage.COMPLETE,
        ArchiveStage.CLEANED,
    ):
        store.advance_archive("L01", stage)

    resumed = store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=1, limit=1))
    assert resumed["work_windows"][-1]["offset"] == 1


def test_gap_before_offset_is_rejected(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    with pytest.raises(ValueError, match="not cleaned"):
        store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=2, limit=1))


def test_cleaned_overlap_replay_is_idempotent(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    # offset=0 never has a predecessor gap, so replaying it is always accepted.
    replay = store.create_or_resume_run(_identity(), ArchiveWorkWindow(offset=0, limit=1))
    assert len(replay["work_windows"]) == 2


# ---------------------------------------------------------------------------
# Archive stage transitions
# ---------------------------------------------------------------------------


def test_archive_transitions_are_ordered_forward_only(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_archive("L01", position=0)
    store.advance_archive("L01", ArchiveStage.DOWNLOADING)
    store.advance_archive("L01", ArchiveStage.DOWNLOADED)
    record = store.get_archive("L01")
    assert record is not None
    assert record.stage == ArchiveStage.DOWNLOADED


def test_archive_transition_replay_is_idempotent(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_archive("L01", position=0)
    store.advance_archive("L01", ArchiveStage.DOWNLOADING)
    store.advance_archive("L01", ArchiveStage.DOWNLOADING)  # no-op replay
    assert store.get_archive("L01").stage == ArchiveStage.DOWNLOADING


def test_archive_transition_rejects_skipped_step(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_archive("L01", position=0)
    with pytest.raises(ValueError, match="not an allowed single forward step"):
        store.advance_archive("L01", ArchiveStage.EXTRACTED)


def test_archive_transition_rejects_reversal(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_archive("L01", position=0)
    store.advance_archive("L01", ArchiveStage.DOWNLOADING)
    store.advance_archive("L01", ArchiveStage.DOWNLOADED)
    with pytest.raises(ValueError, match="not an allowed single forward step"):
        store.advance_archive("L01", ArchiveStage.DOWNLOADING)


# ---------------------------------------------------------------------------
# Batch stage transitions + ceiling
# ---------------------------------------------------------------------------


def test_deterministic_batch_ids_are_stable() -> None:
    assert compute_batch_id("L01", 0) == "L01-batch000"
    assert compute_batch_id("L01", 0) == compute_batch_id("L01", 0)
    assert compute_batch_id("L01", 1) != compute_batch_id("L01", 0)


def test_batch_ceiling_rejects_more_than_eight_videos(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    video_ids = [f"L01_V{i:03d}" for i in range(9)]
    with pytest.raises(ValueError, match="eight-video ceiling"):
        store.ensure_batch("L01-batch000", "L01", video_ids)


def test_batch_transitions_are_ordered_forward_only(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_batch("L01-batch000", "L01", ["L01_V001"])
    store.advance_batch("L01-batch000", BatchStage.EXTRACTED)
    store.advance_batch("L01-batch000", BatchStage.ARTIFACTS_COMPLETE)
    with pytest.raises(ValueError, match="not an allowed single forward step"):
        store.advance_batch("L01-batch000", BatchStage.COMMITTED)


# ---------------------------------------------------------------------------
# Video stage transitions + bounded failure history
# ---------------------------------------------------------------------------


def test_replaying_an_uncommitted_batch_keeps_the_furthest_stage(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_batch("L01-batch000", "L01", ["L01_V001"])
    store.ensure_video("L01_V001", "L01-batch000")
    store.advance_video("L01_V001", VideoStage.SOURCE_READY)
    store.advance_video("L01_V001", VideoStage.EXTRACTED)
    store.advance_batch("L01-batch000", BatchStage.EXTRACTED)

    store.advance_video("L01_V001", VideoStage.SOURCE_READY)
    store.advance_batch("L01-batch000", BatchStage.EXTRACTED)

    assert store.get_video("L01_V001").stage == VideoStage.EXTRACTED
    assert store.get_batch("L01-batch000").stage == BatchStage.EXTRACTED
    store.advance_video("L01_V001", VideoStage.CAPTIONED)
    assert store.get_video("L01_V001").stage == VideoStage.CAPTIONED


def test_video_transitions_are_ordered_forward_only(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_video("L01_V001", "L01-batch000")
    store.advance_video("L01_V001", VideoStage.SOURCE_READY)
    with pytest.raises(ValueError, match="not an allowed single forward step"):
        store.advance_video("L01_V001", VideoStage.CAPTIONED)


def test_failure_history_is_bounded(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_video("L01_V001", "L01-batch000")
    for attempt in range(10):
        store.record_video_failure("L01_V001", {"attempt": attempt, "error": "oom"})
    record = store.get_video("L01_V001")
    assert record is not None
    assert len(record.failures) == 5
    assert record.failures[-1]["attempt"] == 9


# ---------------------------------------------------------------------------
# Ephemeral cleanup guard
# ---------------------------------------------------------------------------


def test_cleanup_is_forbidden_before_local_commit(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_batch("L01-batch000", "L01", ["L01_V001"])
    store.ensure_video("L01_V001", "L01-batch000")
    with pytest.raises(ValueError, match="requires committed or ephemeral_cleaned"):
        store.require_ephemeral_cleanup_allowed("L01-batch000")


def test_cleanup_is_forbidden_when_a_video_is_not_local_complete(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_batch("L01-batch000", "L01", ["L01_V001"])
    store.ensure_video("L01_V001", "L01-batch000")
    for stage in (
        BatchStage.EXTRACTED,
        BatchStage.ARTIFACTS_COMPLETE,
        BatchStage.INDEXES_COMPLETE,
        BatchStage.COMMITTED,
    ):
        store.advance_batch("L01-batch000", stage)
    with pytest.raises(ValueError, match="not local_complete"):
        store.require_ephemeral_cleanup_allowed("L01-batch000")


def test_cleanup_is_allowed_once_committed_and_videos_are_local_complete(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path)
    store.ensure_batch("L01-batch000", "L01", ["L01_V001"])
    store.ensure_video("L01_V001", "L01-batch000")
    for stage in (
        VideoStage.SOURCE_READY,
        VideoStage.EXTRACTED,
        VideoStage.CAPTIONED,
        VideoStage.OCR_COMPLETE,
        VideoStage.OBJECTS_COMPLETE,
        VideoStage.CONTEXT_COMPLETE,
        VideoStage.EMBEDDINGS_COMPLETE,
        VideoStage.LOCAL_COMPLETE,
    ):
        store.advance_video("L01_V001", stage)
    for stage in (
        BatchStage.EXTRACTED,
        BatchStage.ARTIFACTS_COMPLETE,
        BatchStage.INDEXES_COMPLETE,
        BatchStage.COMMITTED,
    ):
        store.advance_batch("L01-batch000", stage)

    record = store.require_ephemeral_cleanup_allowed("L01-batch000")
    assert record.stage == BatchStage.COMMITTED

    store.advance_batch("L01-batch000", BatchStage.EPHEMERAL_CLEANED)
    record = store.require_ephemeral_cleanup_allowed("L01-batch000")
    assert record.stage == BatchStage.EPHEMERAL_CLEANED
