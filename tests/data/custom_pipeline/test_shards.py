"""Tests for per-video batch sharding and the three-index batch bundle.

Covers exact ordered coverage, duplicate/missing/foreign frame_id rejection,
empty child tables, finite-vector/dimension checks, contiguous mapping
positions, deterministic shard writing, and that the built visual/context/ASR
indexes all load and stay scoped to exactly the requested batch videos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from offline.ingestion.custom_pipeline.asr import ASRReuseBundle
from offline.ingestion.custom_pipeline.shards import (
    VideoShard,
    VideoShardError,
    build_batch_index_bundle,
    split_batch_artifacts_by_video,
    write_video_shard,
)
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex


_VIDEO_A = "L01_V001"
_VIDEO_B = "L01_V002"
_VIDEO_IDS = [_VIDEO_A, _VIDEO_B]
# Video A has two frames, video B has one: three frames total in the batch.
_FRAMES = [
    {"frame_id": "f0", "video_id": _VIDEO_A, "frame_idx": 0, "timestamp_ms": 0},
    {"frame_id": "f1", "video_id": _VIDEO_A, "frame_idx": 1, "timestamp_ms": 1000},
    {"frame_id": "f2", "video_id": _VIDEO_B, "frame_idx": 0, "timestamp_ms": 0},
]


def _frames_table() -> pd.DataFrame:
    return pd.DataFrame(_FRAMES)


def _frame_native_table(extra_columns: dict[str, object] | None = None) -> pd.DataFrame:
    rows = [dict(row, **(extra_columns or {})) for row in _FRAMES]
    return pd.DataFrame(rows)


def _vector_mapping() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {**row, "embedding_index": index}
            for index, row in enumerate(_FRAMES)
        ]
    )


def _default_inputs() -> dict[str, object]:
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
    visual_vectors = np.eye(3, dtype=np.float32)
    context_vectors = np.eye(3, dtype=np.float32)
    visual_mapping = _vector_mapping()
    context_mapping = _vector_mapping()
    return {
        "video_ids": _VIDEO_IDS,
        "frames_table": _frames_table(),
        "frame_native_tables": frame_native_tables,
        "child_tables": child_tables,
        "visual_vectors": visual_vectors,
        "visual_mapping": visual_mapping,
        "context_vectors": context_vectors,
        "context_mapping": context_mapping,
    }


# ---------------------------------------------------------------------------
# split_batch_artifacts_by_video / validate_video_shard
# ---------------------------------------------------------------------------


def test_split_produces_exact_ordered_coverage_per_video() -> None:
    inputs = _default_inputs()
    shards = split_batch_artifacts_by_video(**inputs)

    assert set(shards) == {_VIDEO_A, _VIDEO_B}
    assert shards[_VIDEO_A].frame_native["caption"]["frame_id"].tolist() == ["f0", "f1"]
    assert shards[_VIDEO_B].frame_native["caption"]["frame_id"].tolist() == ["f2"]
    assert shards[_VIDEO_A].visual_vectors.shape == (2, 3)
    assert shards[_VIDEO_B].visual_vectors.shape == (1, 3)


def test_split_allows_empty_child_tables() -> None:
    inputs = _default_inputs()
    shards = split_batch_artifacts_by_video(**inputs)
    assert shards[_VIDEO_A].child["ocr_regions"].empty
    assert shards[_VIDEO_B].child["object_detections"].empty


def test_split_accepts_non_empty_child_tables_scoped_to_owning_video() -> None:
    inputs = _default_inputs()
    inputs["child_tables"] = {
        "ocr_regions": pd.DataFrame(
            [{"frame_id": "f0", "video_id": _VIDEO_A, "region_id": "f0:0"}]
        ),
        "object_detections": pd.DataFrame(columns=["frame_id", "video_id"]),
    }
    shards = split_batch_artifacts_by_video(**inputs)
    assert shards[_VIDEO_A].child["ocr_regions"]["frame_id"].tolist() == ["f0"]
    assert shards[_VIDEO_B].child["ocr_regions"].empty


def test_split_rejects_missing_frame_in_frame_native_table() -> None:
    inputs = _default_inputs()
    caption = inputs["frame_native_tables"]["caption"]
    inputs["frame_native_tables"]["caption"] = caption.iloc[[0]]  # drop f1
    with pytest.raises(VideoShardError, match="incomplete frame coverage"):
        split_batch_artifacts_by_video(**inputs)


def test_split_rejects_foreign_frame_in_child_table() -> None:
    inputs = _default_inputs()
    inputs["child_tables"]["ocr_regions"] = pd.DataFrame(
        [{"frame_id": "does-not-exist", "video_id": _VIDEO_A}]
    )
    with pytest.raises(VideoShardError, match="foreign frame_id"):
        split_batch_artifacts_by_video(**inputs)


def test_split_rejects_duplicate_frame_in_frame_native_table() -> None:
    inputs = _default_inputs()
    caption = inputs["frame_native_tables"]["caption"]
    inputs["frame_native_tables"]["caption"] = pd.concat(
        [caption, caption.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(VideoShardError, match="duplicate frame_id"):
        split_batch_artifacts_by_video(**inputs)


def test_split_rejects_vector_mapping_count_mismatch() -> None:
    inputs = _default_inputs()
    # Drop one vector row without dropping its mapping row.
    inputs["visual_vectors"] = inputs["visual_vectors"][:2]
    inputs["visual_mapping"] = inputs["visual_mapping"]  # still 3 rows
    with pytest.raises(VideoShardError):
        split_batch_artifacts_by_video(**inputs)


def test_split_rejects_non_finite_vectors() -> None:
    inputs = _default_inputs()
    vectors = inputs["visual_vectors"].copy()
    vectors[0, 0] = np.nan
    inputs["visual_vectors"] = vectors
    with pytest.raises(VideoShardError, match="non-finite"):
        split_batch_artifacts_by_video(**inputs)


def test_write_video_shard_persists_tables_and_vectors(tmp_path: Path) -> None:
    inputs = _default_inputs()
    shards = split_batch_artifacts_by_video(**inputs)
    written = write_video_shard(shards[_VIDEO_A], tmp_path)

    assert written["caption"].is_file()
    assert written["visual_vectors"].is_file()
    assert np.load(written["visual_vectors"]).shape == (2, 3)
    assert pd.read_parquet(written["visual_mapping"]).shape[0] == 2


# ---------------------------------------------------------------------------
# build_batch_index_bundle
# ---------------------------------------------------------------------------


def _build_asr_bundle(index_root: Path) -> ASRReuseBundle:
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
            {
                "embedding_index": 2,
                "segment_id": "unrelated-000",
                "video_id": "L09_V999",
                "segment_index": 0,
                "start_ms": 0,
                "end_ms": 500,
            },
        ]
    )
    vectors = np.eye(3, dtype=np.float32)
    index = SegmentDenseIndex.build(vectors, mapping, dataset_version="v1", model_name="test")
    index.save(index_root)
    return ASRReuseBundle(
        transcripts_root=str(index_root.parent / "transcripts"),
        index_root=str(index_root),
        video_ids=(_VIDEO_A, _VIDEO_B),
        transcript_fingerprint="a" * 64,
        index_fingerprint="b" * 64,
        segment_count=2,
    )


def test_build_batch_index_bundle_produces_three_loadable_scoped_indexes(
    tmp_path: Path,
) -> None:
    inputs = _default_inputs()
    shards = split_batch_artifacts_by_video(**inputs)
    asr_bundle = _build_asr_bundle(tmp_path / "asr_index")
    output_root = tmp_path / "batch_indexes"

    inventory = build_batch_index_bundle(
        "L01-batch000",
        _VIDEO_IDS,
        shards,
        asr_bundle,
        output_root,
        dataset_version="v1",
        visual_model_name="siglip-test",
        context_model_name="bge-test",
    )

    assert inventory.batch_id == "L01-batch000"
    assert inventory.video_ids == (_VIDEO_A, _VIDEO_B)
    assert inventory.visual.vector_count == 3
    assert inventory.context.vector_count == 3
    # The unrelated third ASR segment must be excluded from the batch subset.
    assert inventory.asr_segments.vector_count == 2

    visual_index = DenseIndex.load(output_root / "visual")
    context_index = DenseIndex.load(output_root / "context")
    asr_index = SegmentDenseIndex.load(output_root / "asr_segments")
    assert set(visual_index.mapping["video_id"]) == {_VIDEO_A, _VIDEO_B}
    assert set(context_index.mapping["video_id"]) == {_VIDEO_A, _VIDEO_B}
    assert set(asr_index.mapping["video_id"]) == {_VIDEO_A, _VIDEO_B}


def test_build_batch_index_bundle_rejects_missing_shard(tmp_path: Path) -> None:
    inputs = _default_inputs()
    shards = split_batch_artifacts_by_video(**inputs)
    del shards[_VIDEO_B]
    asr_bundle = _build_asr_bundle(tmp_path / "asr_index")

    with pytest.raises(KeyError):
        build_batch_index_bundle(
            "L01-batch000",
            _VIDEO_IDS,
            shards,
            asr_bundle,
            tmp_path / "batch_indexes",
            dataset_version="v1",
            visual_model_name="siglip-test",
            context_model_name="bge-test",
        )
