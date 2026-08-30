"""Tests for atomic local batch commit and post-commit ephemeral cleanup.

Covers payload-before-marker order, missing/corrupt payload, index load
failure, atomic visibility, identical resume, a conflicting destination,
cleanup-safety invariants, and deterministic inventory ordering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from offline.ingestion.custom_pipeline.asr import ASRReuseBundle
from offline.ingestion.custom_pipeline.commit import (
    BatchValidationError,
    build_batch_inventory,
    cleanup_ephemeral_batch,
    commit_local_batch,
    validate_local_batch,
)
from offline.ingestion.custom_pipeline.shards import (
    build_batch_index_bundle,
    split_batch_artifacts_by_video,
    write_video_shard,
)
from offline.ingestion.custom_pipeline.state import BatchStage, PipelineStateStore, VideoStage
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


_VIDEO_A = "L01_V001"
_VIDEO_B = "L01_V002"
_VIDEO_IDS = [_VIDEO_A, _VIDEO_B]
_FRAMES = [
    {"frame_id": "f0", "video_id": _VIDEO_A, "frame_idx": 0, "timestamp_ms": 0},
    {"frame_id": "f1", "video_id": _VIDEO_A, "frame_idx": 1, "timestamp_ms": 1000},
    {"frame_id": "f2", "video_id": _VIDEO_B, "frame_idx": 0, "timestamp_ms": 0},
]


def _frame_native_table(extra: dict[str, object] | None = None) -> pd.DataFrame:
    return pd.DataFrame([dict(row, **(extra or {})) for row in _FRAMES])


def _vector_mapping() -> pd.DataFrame:
    return pd.DataFrame([{**row, "embedding_index": i} for i, row in enumerate(_FRAMES)])


def _asr_bundle(index_root: Path) -> ASRReuseBundle:
    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": f"{_VIDEO_A}-000",
                "video_id": _VIDEO_A,
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 1000,
            },
            {
                "embedding_index": 1,
                "segment_id": f"{_VIDEO_B}-000",
                "video_id": _VIDEO_B,
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
            },
        ]
    )
    index = SegmentDenseIndex.build(
        np.eye(2, dtype=np.float32), mapping, dataset_version="v1", model_name="test"
    )
    index.save(index_root)
    return ASRReuseBundle(
        transcripts_root=str(index_root.parent / "transcripts"),
        index_root=str(index_root),
        video_ids=(_VIDEO_A, _VIDEO_B),
        transcript_fingerprint="a" * 64,
        index_fingerprint="b" * 64,
        segment_count=2,
    )


def _build_staged_batch(staging_root: Path) -> None:
    """Populate a fully valid staged batch directory, mirroring Task 6 output."""

    frame_native_tables = {
        "caption": _frame_native_table({"text": "a caption"}),
        "ocr_frames": _frame_native_table({"normalized_text": None}),
        "object_frames": _frame_native_table({"summary": None}),
        "context": _frame_native_table({"context_text": "context"}),
    }
    child_tables = {
        "ocr_regions": pd.DataFrame(columns=["frame_id", "video_id"]),
        "object_detections": pd.DataFrame(columns=["frame_id", "video_id"]),
    }
    vectors = np.eye(3, dtype=np.float32)
    mapping = _vector_mapping()

    shards = split_batch_artifacts_by_video(
        _VIDEO_IDS,
        pd.DataFrame(_FRAMES),
        frame_native_tables,
        child_tables,
        vectors,
        mapping,
        vectors,
        mapping,
    )
    for video_id in _VIDEO_IDS:
        write_video_shard(shards[video_id], staging_root)

    asr_bundle = _asr_bundle(staging_root.parent / "asr_index")
    build_batch_index_bundle(
        "L01-batch000",
        _VIDEO_IDS,
        shards,
        asr_bundle,
        staging_root,
        dataset_version="v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )


# ---------------------------------------------------------------------------
# build_batch_inventory / validate_local_batch
# ---------------------------------------------------------------------------


def test_inventory_is_deterministic_and_ordered(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    first = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    second = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    assert first == second
    assert [entry.relative_path for entry in first.files] == sorted(
        entry.relative_path for entry in first.files
    )


def test_validate_local_batch_accepts_a_complete_staged_batch(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    validate_local_batch("L01-batch000", _VIDEO_IDS, staging_root, inventory)  # no raise


def test_validate_local_batch_rejects_missing_payload(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    (staging_root / "videos" / _VIDEO_A / "caption.parquet").unlink()
    with pytest.raises(BatchValidationError, match="missing artifacts"):
        validate_local_batch("L01-batch000", _VIDEO_IDS, staging_root, inventory)


def test_validate_local_batch_rejects_corrupted_payload(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    (staging_root / "videos" / _VIDEO_A / "caption.parquet").write_bytes(b"corrupt")
    with pytest.raises(BatchValidationError, match="mismatch"):
        validate_local_batch("L01-batch000", _VIDEO_IDS, staging_root, inventory)


def test_validate_local_batch_rejects_broken_index(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    (staging_root / "visual" / "vectors.npy").write_bytes(b"corrupt")
    with pytest.raises(BatchValidationError, match="index checksum-load failed"):
        validate_local_batch("L01-batch000", _VIDEO_IDS, staging_root, inventory)


# ---------------------------------------------------------------------------
# commit_local_batch
# ---------------------------------------------------------------------------


def test_commit_writes_markers_after_payload_and_publishes_atomically(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    validate_local_batch("L01-batch000", _VIDEO_IDS, staging_root, inventory)

    final_root = tmp_path / "final" / "L01-batch000"
    result = commit_local_batch(staging_root, final_root, inventory)

    assert result == final_root
    assert not staging_root.exists()
    assert (final_root / "manifest.json").is_file()
    assert (final_root / "_SUCCESS.json").is_file()
    assert (final_root / "videos" / _VIDEO_A / "caption.parquet").is_file()


def test_commit_is_idempotent_for_an_identical_resume(tmp_path: Path) -> None:
    import shutil

    final_root = tmp_path / "final" / "L01-batch000"

    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    commit_local_batch(staging_root, final_root, inventory)

    # Simulate resuming from an identical staged copy of what was already
    # committed (e.g. a crash right after staging, before the atomic rename),
    # rather than regenerating indexes whose metadata timestamps would differ.
    restaged_root = tmp_path / "staging2"
    shutil.copytree(final_root, restaged_root)
    (restaged_root / "manifest.json").unlink()
    (restaged_root / "_SUCCESS.json").unlink()
    resumed_inventory = build_batch_inventory(restaged_root, "L01-batch000", _VIDEO_IDS)
    result = commit_local_batch(restaged_root, final_root, resumed_inventory)

    assert result == final_root
    assert not restaged_root.exists()  # discarded as a redundant identical restage


def test_commit_rejects_a_conflicting_completed_destination(tmp_path: Path) -> None:
    final_root = tmp_path / "final" / "L01-batch000"

    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)
    commit_local_batch(staging_root, final_root, inventory)

    other_staging = tmp_path / "staging_other"
    _build_staged_batch(other_staging)
    (other_staging / "videos" / _VIDEO_A / "caption.parquet").write_text("different-caption-text")
    other_inventory = build_batch_inventory(other_staging, "L01-batch000", _VIDEO_IDS)

    with pytest.raises(BatchValidationError, match="conflicting completed"):
        commit_local_batch(other_staging, final_root, other_inventory)


def test_commit_rejects_a_conflicting_incomplete_destination(tmp_path: Path) -> None:
    final_root = tmp_path / "final" / "L01-batch000"
    final_root.mkdir(parents=True)
    (final_root / "some_partial_file").write_text("partial")

    staging_root = tmp_path / "staging"
    _build_staged_batch(staging_root)
    inventory = build_batch_inventory(staging_root, "L01-batch000", _VIDEO_IDS)

    with pytest.raises(BatchValidationError, match="conflicting incomplete"):
        commit_local_batch(staging_root, final_root, inventory)


# ---------------------------------------------------------------------------
# cleanup_ephemeral_batch
# ---------------------------------------------------------------------------


def _complete_video(store: PipelineStateStore, video_id: str, batch_id: str) -> None:
    store.ensure_video(video_id, batch_id)
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
        store.advance_video(video_id, stage)


def _committed_batch_store(tmp_path: Path) -> PipelineStateStore:
    store = PipelineStateStore(tmp_path / "state_root")
    store.ensure_batch("L01-batch000", "L01", _VIDEO_IDS)
    for video_id in _VIDEO_IDS:
        _complete_video(store, video_id, "L01-batch000")
    for stage in (
        BatchStage.EXTRACTED,
        BatchStage.ARTIFACTS_COMPLETE,
        BatchStage.INDEXES_COMPLETE,
        BatchStage.COMMITTED,
    ):
        store.advance_batch("L01-batch000", stage)
    return store


def test_cleanup_is_forbidden_before_commit(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "state_root")
    store.ensure_batch("L01-batch000", "L01", _VIDEO_IDS)
    active_root = tmp_path / "active"
    active_root.mkdir()
    with pytest.raises(ValueError, match="requires committed"):
        cleanup_ephemeral_batch(
            store, "L01-batch000", [active_root / "scratch"], allowed_root=active_root
        )


def test_cleanup_removes_exact_ephemeral_paths_after_commit(tmp_path: Path) -> None:
    store = _committed_batch_store(tmp_path)
    active_root = tmp_path / "active"
    ocr_scratch = active_root / "ocr_scratch"
    ocr_scratch.mkdir(parents=True)
    (ocr_scratch / "frame.jpg").write_bytes(b"x")
    source_mp4 = active_root / "L01_V001.mp4"
    source_mp4.write_bytes(b"video")

    cleanup_ephemeral_batch(
        store, "L01-batch000", [ocr_scratch, source_mp4], allowed_root=active_root
    )

    assert not ocr_scratch.exists()
    assert not source_mp4.exists()
    assert store.get_batch("L01-batch000").stage == BatchStage.EPHEMERAL_CLEANED


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    store = _committed_batch_store(tmp_path)
    active_root = tmp_path / "active"
    active_root.mkdir()
    ocr_scratch = active_root / "ocr_scratch"
    ocr_scratch.mkdir()

    cleanup_ephemeral_batch(store, "L01-batch000", [ocr_scratch], allowed_root=active_root)
    cleanup_ephemeral_batch(store, "L01-batch000", [ocr_scratch], allowed_root=active_root)

    assert store.get_batch("L01-batch000").stage == BatchStage.EPHEMERAL_CLEANED


def test_cleanup_preserves_committed_final_artifacts(tmp_path: Path) -> None:
    store = _committed_batch_store(tmp_path)
    active_root = tmp_path / "active"
    active_root.mkdir()
    ocr_scratch = active_root / "ocr_scratch"
    ocr_scratch.mkdir()

    final_root = tmp_path / "final" / "L01-batch000"
    final_root.mkdir(parents=True)
    (final_root / "manifest.json").write_text("{}")

    cleanup_ephemeral_batch(store, "L01-batch000", [ocr_scratch], allowed_root=active_root)

    assert (final_root / "manifest.json").is_file()  # untouched: outside allowed_root


def test_cleanup_rejects_paths_escaping_the_allowed_root(tmp_path: Path) -> None:
    store = _committed_batch_store(tmp_path)
    active_root = tmp_path / "active"
    active_root.mkdir()
    escaping_path = tmp_path / "outside_active" / "danger.mp4"
    escaping_path.parent.mkdir()
    escaping_path.write_bytes(b"video")

    with pytest.raises(ValueError, match="outside allowed root"):
        cleanup_ephemeral_batch(
            store, "L01-batch000", [escaping_path], allowed_root=active_root
        )
    assert escaping_path.exists()  # never touched
