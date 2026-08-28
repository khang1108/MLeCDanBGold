"""Tests for runner composition: preflight, one archive, status, finalize.

Uses a fake curl download (monkeypatched subprocess), a synthetic ZIP archive
of two videos, and fake ``produce_batch_artifacts``/``asr_bundle_factory``
callbacks so the full archive-to-committed-corpus path is exercised without
any real model, GPU, or network dependency.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.data.custom_pipeline.asr import ASRReuseBundle
from hcmai.data.custom_pipeline.config import (
    ArchivePlan,
    ArchiveWorkWindow,
    DiskBudgetConfig,
    SchedulingConfig,
)
from hcmai.data.custom_pipeline.runner import (
    BatchArtifacts,
    RunnerContext,
    finalize_pipeline,
    pipeline_status,
    preflight_pipeline,
    process_archive,
)
from hcmai.data.custom_pipeline.state import ArchiveStage, PipelineStateStore, VideoStage
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


_VIDEO_A = "L01_V001"
_VIDEO_B = "L01_V002"


def _generous_budget() -> DiskBudgetConfig:
    return DiskBudgetConfig(
        min_free_gib=0.000001,
        max_active_gib=1,
        max_archive_download_gib=1,
        max_archive_uncompressed_gib=1,
    )


def _write_fake_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("L01/L01_V001.mp4", b"a" * 100)
        archive.writestr("L01/L01_V002.mp4", b"b" * 100)


@pytest.fixture()
def fake_curl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Replace ``subprocess.run`` so ``download_archive`` copies a fixed ZIP."""

    source_zip = tmp_path / "_source.zip"
    _write_fake_zip(source_zip)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        part_path = Path(argv[argv.index("-o") + 1])
        shutil.copyfile(source_zip, part_path)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return source_zip


def _asr_bundle_factory(index_root: Path) -> callable:
    def _factory(video_ids: list[str]) -> ASRReuseBundle:
        mapping = pd.DataFrame(
            [
                {
                    "embedding_index": position,
                    "segment_id": f"{video_id}-000",
                    "video_id": video_id,
                    "segment_index": 0,
                    "start_ms": 0,
                    "end_ms": 1000,
                }
                for position, video_id in enumerate(video_ids)
            ]
        )
        index = SegmentDenseIndex.build(
            np.eye(len(video_ids), dtype=np.float32), mapping, dataset_version="v1", model_name="asr-test"
        )
        index.save(index_root / "-".join(video_ids))
        return ASRReuseBundle(
            transcripts_root=str(index_root / "transcripts"),
            index_root=str(index_root / "-".join(video_ids)),
            video_ids=tuple(video_ids),
            transcript_fingerprint="a" * 64,
            index_fingerprint="b" * 64,
            segment_count=len(video_ids),
        )

    return _factory


def _make_produce_batch_artifacts(state_store: PipelineStateStore):
    def _produce(batch_id: str, video_ids: list[str], source_paths: list[Path]) -> BatchArtifacts:
        frames = [
            {"frame_id": f"{video_id}_f0", "video_id": video_id, "frame_idx": 0, "timestamp_ms": 0}
            for video_id in video_ids
        ]
        frames_table = pd.DataFrame(frames)
        frame_native_tables = {
            "caption": pd.DataFrame([dict(row, text="a caption") for row in frames]),
            "ocr_frames": pd.DataFrame([dict(row, normalized_text=None) for row in frames]),
            "object_frames": pd.DataFrame([dict(row, summary=None) for row in frames]),
            "context": pd.DataFrame([dict(row, context_text="context") for row in frames]),
        }
        child_tables = {
            "ocr_regions": pd.DataFrame(columns=["frame_id", "video_id"]),
            "object_detections": pd.DataFrame(columns=["frame_id", "video_id"]),
        }
        mapping = pd.DataFrame([{**row, "embedding_index": i} for i, row in enumerate(frames)])
        vectors = np.random.default_rng(0).standard_normal((len(frames), 4)).astype(np.float32)

        for video_id in video_ids:
            for stage in (
                VideoStage.SOURCE_READY,
                VideoStage.EXTRACTED,
                VideoStage.CAPTIONED,
                VideoStage.OCR_COMPLETE,
                VideoStage.OBJECTS_COMPLETE,
                VideoStage.CONTEXT_COMPLETE,
                VideoStage.EMBEDDINGS_COMPLETE,
            ):
                state_store.advance_video(video_id, stage)

        return BatchArtifacts(
            frames_table=frames_table,
            frame_native_tables=frame_native_tables,
            child_tables=child_tables,
            visual_vectors=vectors,
            visual_mapping=mapping,
            context_vectors=vectors,
            context_mapping=mapping,
        )

    return _produce


def _context(tmp_path: Path) -> RunnerContext:
    native_executable = tmp_path / "keyframe_extractor"
    native_executable.write_text("fake")
    return RunnerContext(
        run_root=tmp_path / "runs" / "dataset_v1",
        artifacts_root=tmp_path / "artifacts" / "dataset_v1",
        native_executable=native_executable,
        disk_budget=_generous_budget(),
        scheduling=SchedulingConfig(),
    )


# ---------------------------------------------------------------------------
# preflight_pipeline
# ---------------------------------------------------------------------------


def test_preflight_reports_ok_with_real_native_ffmpeg_curl(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    window = ArchiveWorkWindow()

    report = preflight_pipeline(context, plan, window)

    assert report.native_executable_found
    assert report.ffmpeg_found
    assert report.curl_found
    assert report.archive_plan_size == 1


def test_preflight_reports_problems_without_downloading(tmp_path: Path) -> None:
    context = RunnerContext(
        run_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
        native_executable=tmp_path / "missing_executable",
        disk_budget=_generous_budget(),
        scheduling=SchedulingConfig(),
    )
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    report = preflight_pipeline(context, plan, ArchiveWorkWindow(offset=5))

    assert not report.ok
    assert not report.native_executable_found
    assert any("out of range" in problem for problem in report.problems)
    assert not (tmp_path / "runs" / "active" / "archives").exists()


# ---------------------------------------------------------------------------
# process_archive / pipeline_status / finalize_pipeline
# ---------------------------------------------------------------------------


def test_process_archive_commits_one_batch_and_cleans_archive(
    fake_curl: Path, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(
        _identity(plan), ArchiveWorkWindow(offset=0, limit=1)
    )

    produce = _make_produce_batch_artifacts(state_store)
    asr_factory = _asr_bundle_factory(tmp_path / "asr_indexes")

    batch_ids = process_archive(
        context,
        state_store,
        plan.entries[0],
        produce,
        asr_factory,
        dataset_version="dataset_v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )

    assert len(batch_ids) == 1
    final_batch_root = context.artifacts_root / "batches" / plan.entries[0].archive_id / batch_ids[0]
    assert (final_batch_root / "manifest.json").is_file()
    assert (final_batch_root / "_SUCCESS.json").is_file()

    archive_record = state_store.get_archive(plan.entries[0].archive_id)
    assert archive_record is not None and archive_record.stage == ArchiveStage.CLEANED
    assert state_store.get_video(_VIDEO_A).stage == VideoStage.LOCAL_COMPLETE
    assert state_store.get_video(_VIDEO_B).stage == VideoStage.LOCAL_COMPLETE
    # The extracted archive directory itself must be gone after cleanup.
    assert not (context.active_root / "archives" / plan.entries[0].archive_id).exists()


def test_process_archive_resumes_an_interrupted_download(
    fake_curl: Path, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(
        _identity(plan), ArchiveWorkWindow(offset=0, limit=1)
    )
    archive_id = plan.entries[0].archive_id
    state_store.ensure_archive(archive_id, plan.entries[0].position)
    state_store.advance_archive(archive_id, ArchiveStage.DOWNLOADING)

    batch_ids = process_archive(
        context,
        state_store,
        plan.entries[0],
        _make_produce_batch_artifacts(state_store),
        _asr_bundle_factory(tmp_path / "asr_indexes"),
        dataset_version="dataset_v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )

    assert len(batch_ids) == 1
    archive_record = state_store.get_archive(archive_id)
    assert archive_record is not None and archive_record.stage == ArchiveStage.CLEANED


def test_process_archive_commits_only_the_requested_batch_slice(
    fake_curl: Path, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    context = RunnerContext(
        run_root=context.run_root,
        artifacts_root=context.artifacts_root,
        native_executable=context.native_executable,
        disk_budget=context.disk_budget,
        scheduling=SchedulingConfig(max_videos_per_batch=1),
    )
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(_identity(plan), ArchiveWorkWindow(offset=0, limit=1))

    batch_ids = process_archive(
        context,
        state_store,
        plan.entries[0],
        _make_produce_batch_artifacts(state_store),
        _asr_bundle_factory(tmp_path / "asr_indexes"),
        dataset_version="dataset_v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
        batch_offset=1,
        batch_limit=1,
    )

    assert batch_ids == ["Videos_L01-batch001"]
    committed = context.artifacts_root / "batches" / plan.entries[0].archive_id
    assert (committed / "Videos_L01-batch001" / "_SUCCESS.json").is_file()
    assert not (committed / "Videos_L01-batch000").exists()


def test_pipeline_status_reports_recommended_next_offset(fake_curl: Path, tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(_identity(plan), ArchiveWorkWindow(offset=0, limit=1))
    process_archive(
        context,
        state_store,
        plan.entries[0],
        _make_produce_batch_artifacts(state_store),
        _asr_bundle_factory(tmp_path / "asr_indexes"),
        dataset_version="dataset_v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )

    status = pipeline_status(context, state_store, plan)
    assert status["recommended_next_offset"] == 1
    assert status["complete_corpus"] is True
    assert status["archives"][plan.entries[0].archive_id] == "cleaned"


def test_finalize_pipeline_produces_a_report(fake_curl: Path, tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = ArchivePlan.from_urls(["https://example.org/Videos_L01.zip"])
    state_store = PipelineStateStore(context.run_root)
    state_store.create_or_resume_run(_identity(plan), ArchiveWorkWindow(offset=0, limit=1))
    process_archive(
        context,
        state_store,
        plan.entries[0],
        _make_produce_batch_artifacts(state_store),
        _asr_bundle_factory(tmp_path / "asr_indexes"),
        dataset_version="dataset_v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )

    report = finalize_pipeline(
        state_store,
        plan,
        context.artifacts_root / "batches",
        tmp_path / "final_corpus",
        dataset_version="dataset_v1",
    )
    assert report["batch_count"] == 1
    assert report["video_count"] == 2


def _identity(plan: ArchivePlan):
    from hcmai.data.custom_pipeline.contracts import RunIdentity

    return RunIdentity(
        version="dataset_v1",
        source="custom_raw_video_1fps",
        frame_store_id="dataset_1",
        media_info_digest="a" * 64,
        archive_plan_digest=plan.digest,
        artifact_config_fingerprint="c" * 64,
        model_revisions={"caption": "rev1"},
        asr_lineage_digest="d" * 64,
    )
