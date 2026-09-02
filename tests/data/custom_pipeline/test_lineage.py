"""Regression coverage for custom-pipeline embedding lineage propagation.

These tests use small synthetic indexes to verify future custom builds retain
the encoder model name and revision without changing vectors or identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")

from hcmai.common.config import EncoderConfig
from hcmai.common.utils.io import write_json
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
from offline.enrichment.transcripts.manifest import TranscriptManifest
from offline.ingestion.custom_pipeline.asr import ASRReuseBundle, validate_asr_source
from offline.ingestion.custom_pipeline.finalize import (
    BatchManifest,
    FinalizeError,
    build_dense_index_from_precomputed,
    build_segment_index_from_precomputed,
    compact_batch_embeddings,
    compact_batch_embeddings_to_memmap,
)
from offline.ingestion.custom_pipeline.shards import (
    build_batch_index_bundle,
    split_batch_artifacts_by_video,
)


_VIDEO_ID = "L01_V001"


def _frame_mapping(video_id: str = _VIDEO_ID) -> pd.DataFrame:
    """Create one canonical frame mapping for a minimal dense index."""

    return pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "frame_id": f"{video_id}-f000",
                "video_id": video_id,
                "frame_idx": 12,
                "timestamp_ms": 12_000,
            }
        ]
    )


def _segment_mapping(video_id: str = _VIDEO_ID) -> pd.DataFrame:
    """Create one segment-native mapping with no frame identity fields."""

    return pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": f"{video_id}-s000",
                "video_id": video_id,
                "segment_index": 0,
                "start_ms": 12_000,
                "end_ms": 13_000,
            }
        ]
    )


def _asr_bundle(
    tmp_path: Path,
    *,
    model_name: str = "test/evidence-encoder",
    model_revision: str | None = "evidence-revision",
) -> ASRReuseBundle:
    """Persist a reusable ASR source whose lineage is known to the test."""

    index_root = tmp_path / "asr-source-index"
    SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _segment_mapping(),
        dataset_version="source-v1",
        model_name=model_name,
        model_revision=model_revision,
    ).save(index_root)
    return ASRReuseBundle(
        transcripts_root=str(tmp_path / "transcripts"),
        index_root=str(index_root),
        video_ids=(_VIDEO_ID,),
        transcript_fingerprint="a" * 64,
        index_fingerprint="b" * 64,
        segment_count=1,
    )


def _video_shards() -> dict[str, object]:
    """Create one complete video shard with aligned visual/context vectors."""

    frames = _frame_mapping().drop(columns="embedding_index")
    frame_native_tables = {
        "caption": frames.assign(text="caption"),
        "ocr_frames": frames.assign(normalized_text="ocr"),
        "object_frames": frames.assign(summary="object"),
        "context": frames.assign(context_text="context"),
    }
    child_tables = {
        "ocr_regions": pd.DataFrame(columns=["frame_id", "video_id"]),
        "object_detections": pd.DataFrame(columns=["frame_id", "video_id"]),
    }
    mapping = _frame_mapping()
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    return split_batch_artifacts_by_video(
        [_VIDEO_ID],
        frames,
        frame_native_tables,
        child_tables,
        vectors,
        mapping,
        vectors,
        mapping,
    )


def _write_transcript_fixture(tmp_path: Path, index_root: Path) -> tuple[Path, Path]:
    """Write a completed transcript and compatible reusable ASR index."""

    transcripts_root = tmp_path / "transcripts"
    video_root = transcripts_root / "L01"
    video_root.mkdir(parents=True)
    manifest = TranscriptManifest(
        video_id=_VIDEO_ID,
        source={"size_bytes": 1, "sha256": "a" * 64},
        config_sha256="b" * 64,
        asr_model="test/asr",
        asr_revision="c" * 40,
        diarization_enabled=False,
        diarization_model=None,
        diarization_revision=None,
        schema_version="transcript-segment-v1",
        pipeline_version="transcript-pipeline-v1",
        segment_count=1,
        status="completed",
        failure_category=None,
    )
    write_json(manifest.model_dump(mode="json"), video_root / f"{_VIDEO_ID}.manifest.json")
    _segment_mapping().drop(columns="embedding_index").to_parquet(
        video_root / f"{_VIDEO_ID}.parquet", index=False
    )
    SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _segment_mapping(),
        dataset_version="source-v1",
        model_name="test/evidence-encoder",
        model_revision="source-revision",
    ).save(index_root)
    return transcripts_root, index_root


def _batch_manifest(
    tmp_path: Path,
    *,
    batch_id: str,
    video_id: str,
    model_revision: str | None,
) -> BatchManifest:
    """Create one valid batch root containing a visual index only."""

    root = tmp_path / batch_id
    DenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _frame_mapping(video_id),
        dataset_version="batch-v1",
        model_name="test/visual-encoder",
        model_revision=model_revision,
    ).save(root / "visual")
    return BatchManifest(
        batch_id=batch_id,
        root=root,
        video_ids=(video_id,),
        canonical_frame_digest="unused-by-compaction",
    )


def test_index_builders_persist_optional_model_revision() -> None:
    """Dense builders preserve supplied revisions while old calls remain optional."""

    dense = DenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _frame_mapping(),
        dataset_version="v1",
        model_name="test/visual-encoder",
        model_revision="visual-revision",
    )
    segment = SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _segment_mapping(),
        dataset_version="v1",
        model_name="test/evidence-encoder",
        model_revision="evidence-revision",
    )

    assert dense.metadata.model_revision == "visual-revision"
    assert segment.metadata.model_revision == "evidence-revision"


def test_batch_bundle_preserves_configured_and_reused_embedding_lineage(
    tmp_path: Path,
) -> None:
    """Batch indexes retain Visual/Context config and source ASR lineage."""

    build_batch_index_bundle(
        "L01-batch000",
        [_VIDEO_ID],
        _video_shards(),
        _asr_bundle(tmp_path),
        tmp_path / "batch",
        dataset_version="batch-v1",
        visual_model_name="test/visual-encoder",
        visual_model_revision="visual-revision",
        context_model_name="test/evidence-encoder",
        context_model_revision="evidence-revision",
    )

    visual = DenseIndex.load(tmp_path / "batch" / "visual")
    context = DenseIndex.load(tmp_path / "batch" / "context")
    asr = SegmentDenseIndex.load(tmp_path / "batch" / "asr_segments")
    assert (visual.metadata.model_name, visual.metadata.model_revision) == (
        "test/visual-encoder",
        "visual-revision",
    )
    assert (context.metadata.model_name, context.metadata.model_revision) == (
        "test/evidence-encoder",
        "evidence-revision",
    )
    assert (asr.metadata.model_name, asr.metadata.model_revision) == (
        "test/evidence-encoder",
        "evidence-revision",
    )


@pytest.mark.parametrize(
    ("configured_model", "configured_revision", "message"),
    [
        ("other/evidence-encoder", "source-revision", "model differs"),
        ("test/evidence-encoder", "other-revision", "revision differs"),
    ],
)
def test_validate_asr_source_rejects_evidence_encoder_lineage_mismatch(
    tmp_path: Path,
    configured_model: str,
    configured_revision: str,
    message: str,
) -> None:
    """A mismatched reusable ASR source is rejected before batch construction."""

    transcripts_root, index_root = _write_transcript_fixture(tmp_path, tmp_path / "asr-index")
    evidence_encoder = EncoderConfig(
        backend="bge_m3",
        model_name=configured_model,
        revision=configured_revision,
    )

    with pytest.raises(ValueError, match=message):
        validate_asr_source(
            transcripts_root,
            index_root,
            [_VIDEO_ID],
            evidence_encoder=evidence_encoder,
        )


def test_legacy_compaction_preserves_and_validates_model_revision(tmp_path: Path) -> None:
    """The legacy in-memory compactor returns a common revision and rejects drift."""

    first = _batch_manifest(
        tmp_path,
        batch_id="L01-batch000",
        video_id="L01_V001",
        model_revision="visual-revision",
    )
    second = _batch_manifest(
        tmp_path,
        batch_id="L02-batch000",
        video_id="L02_V001",
        model_revision="visual-revision",
    )

    vectors, mapping, model_name, model_revision = compact_batch_embeddings(
        [first, second], "visual"
    )
    assert len(vectors) == len(mapping) == 2
    assert (model_name, model_revision) == ("test/visual-encoder", "visual-revision")

    mismatched = _batch_manifest(
        tmp_path,
        batch_id="L03-batch000",
        video_id="L03_V001",
        model_revision="other-revision",
    )
    with pytest.raises(FinalizeError, match="model_revision"):
        compact_batch_embeddings([first, mismatched], "visual")
    with pytest.raises(FinalizeError, match="model_revision"):
        compact_batch_embeddings_to_memmap(
            [first, mismatched],
            "visual",
            tmp_path / "vectors.npy",
            batch_chunk_size=1,
        )


def test_legacy_finalize_builders_persist_model_revision(tmp_path: Path) -> None:
    """Legacy precomputed-vector helpers propagate optional revisions to indexes."""

    dense = build_dense_index_from_precomputed(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _frame_mapping(),
        tmp_path / "visual",
        dataset_version="v1",
        model_name="test/visual-encoder",
        model_revision="visual-revision",
    )
    segment = build_segment_index_from_precomputed(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _segment_mapping(),
        tmp_path / "asr",
        dataset_version="v1",
        model_name="test/evidence-encoder",
        model_revision="evidence-revision",
    )

    assert dense.metadata.model_revision == "visual-revision"
    assert segment.metadata.model_revision == "evidence-revision"
