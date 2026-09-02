"""Regression coverage for custom-pipeline embedding lineage propagation.

These tests use small synthetic indexes to verify future custom builds retain
the encoder model name and revision without changing vectors or identity.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    build_index_from_chunked_embeddings,
    build_segment_index_from_precomputed,
    compact_batch_embeddings,
    compact_batch_embeddings_to_memmap,
)
from offline.ingestion.custom_pipeline.shards import (
    build_batch_index_bundle,
    split_batch_artifacts_by_video,
)
from offline.ingestion.custom_pipeline import runner
from scripts import prepare_custom_pipeline as prepare_pipeline


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
    source_fingerprint: str | None = "c" * 64,
    config_fingerprint: str | None = "d" * 64,
) -> ASRReuseBundle:
    """Persist a reusable ASR source whose lineage is known to the test."""

    index_root = tmp_path / "asr-source-index"
    source_index = SegmentDenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        _segment_mapping(),
        dataset_version="source-v1",
        model_name=model_name,
        model_revision=model_revision,
    )
    source_index.metadata.source_fingerprint = source_fingerprint
    source_index.metadata.config_fingerprint = config_fingerprint
    source_index.save(index_root)
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

    asr_bundle = _asr_bundle(tmp_path)
    build_batch_index_bundle(
        "L01-batch000",
        [_VIDEO_ID],
        _video_shards(),
        asr_bundle,
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
    assert asr.metadata.source_fingerprint == "c" * 64
    assert asr.metadata.config_fingerprint == "d" * 64
    assert asr.metadata.source_fingerprint != asr_bundle.transcript_fingerprint
    assert asr.metadata.config_fingerprint != asr_bundle.index_fingerprint


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


def test_legacy_compaction_keeps_its_public_result_shape_and_validates_revision(
    tmp_path: Path,
) -> None:
    """Legacy callers keep a three-tuple while revision drift remains rejected."""

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

    vectors, mapping, model_name = compact_batch_embeddings([first, second], "visual")
    assert len(vectors) == len(mapping) == 2
    assert model_name == "test/visual-encoder"

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


def test_active_finalization_preserves_source_and_config_fingerprints(
    tmp_path: Path,
) -> None:
    """Global ASR output preserves the source index's semantic fingerprints."""

    batch_root = tmp_path / "batch"
    build_batch_index_bundle(
        "L01-batch000",
        [_VIDEO_ID],
        _video_shards(),
        _asr_bundle(tmp_path / "source"),
        batch_root,
        dataset_version="batch-v1",
        visual_model_name="test/visual-encoder",
        visual_model_revision="visual-revision",
        context_model_name="test/evidence-encoder",
        context_model_revision="evidence-revision",
    )
    manifest = BatchManifest(
        batch_id="L01-batch000",
        root=batch_root,
        video_ids=(_VIDEO_ID,),
        canonical_frame_digest="unused-by-compaction",
    )

    compacted = compact_batch_embeddings_to_memmap(
        [manifest],
        "asr_segments",
        tmp_path / "asr-vectors.npy",
        batch_chunk_size=1,
    )
    build_index_from_chunked_embeddings(
        compacted,
        tmp_path / "global-asr",
        dataset_version="global-v1",
        retrieval_source="asr",
    )

    global_asr = SegmentDenseIndex.load(tmp_path / "global-asr")
    assert global_asr.metadata.model_revision == "evidence-revision"
    assert global_asr.metadata.source_fingerprint == "c" * 64
    assert global_asr.metadata.config_fingerprint == "d" * 64


def test_active_finalization_rejects_asr_fingerprint_drift(tmp_path: Path) -> None:
    """Batch ASR metadata must agree before global compaction can proceed."""

    first_root = tmp_path / "first-batch"
    second_root = tmp_path / "second-batch"
    common = {
        "dataset_version": "batch-v1",
        "visual_model_name": "test/visual-encoder",
        "visual_model_revision": "visual-revision",
        "context_model_name": "test/evidence-encoder",
        "context_model_revision": "evidence-revision",
    }
    build_batch_index_bundle(
        "L01-batch000",
        [_VIDEO_ID],
        _video_shards(),
        _asr_bundle(tmp_path / "first-source", config_fingerprint="b" * 64),
        first_root,
        **common,
    )
    build_batch_index_bundle(
        "L02-batch000",
        [_VIDEO_ID],
        _video_shards(),
        _asr_bundle(tmp_path / "second-source", config_fingerprint="c" * 64),
        second_root,
        **common,
    )

    manifests = [
        BatchManifest("L01-batch000", first_root, (_VIDEO_ID,), "unused"),
        BatchManifest("L02-batch000", second_root, (_VIDEO_ID,), "unused"),
    ]
    with pytest.raises(FinalizeError, match="config_fingerprint"):
        compact_batch_embeddings_to_memmap(
            manifests,
            "asr_segments",
            tmp_path / "asr-vectors.npy",
            batch_chunk_size=1,
        )


def test_factory_and_runner_thread_configured_revisions_to_batch_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI factory and runner preserve config lineage at the batch boundary."""

    evidence_encoder = EncoderConfig(
        backend="bge_m3",
        model_name="test/evidence-encoder",
        revision="evidence-revision",
    )
    captured_factory: dict[str, object] = {}
    bundle = _asr_bundle(tmp_path / "factory-source")
    monkeypatch.setattr(prepare_pipeline, "_load_encoder_config", lambda *_: evidence_encoder)
    monkeypatch.setattr(
        prepare_pipeline,
        "validate_asr_source",
        lambda *args, **kwargs: captured_factory.update(kwargs) or bundle,
    )
    factory = prepare_pipeline._make_asr_bundle_factory(
        SimpleNamespace(
            config=tmp_path / "prepare.yaml",
            transcripts_root=tmp_path / "transcripts",
            asr_index_root=tmp_path / "asr-index",
        )
    )
    assert factory((_VIDEO_ID,)) is bundle
    assert captured_factory["evidence_encoder"] == evidence_encoder

    captured_build: dict[str, object] = {}
    monkeypatch.setattr(runner, "stage_archive_source_links", lambda *_: [tmp_path / "video.mp4"])
    monkeypatch.setattr(runner, "split_batch_artifacts_by_video", lambda *_: {_VIDEO_ID: object()})
    monkeypatch.setattr(runner, "write_video_shard", lambda *_: None)
    monkeypatch.setattr(runner, "require_asr_video_coverage", lambda *_: None)
    monkeypatch.setattr(
        runner,
        "build_batch_index_bundle",
        lambda *args, **kwargs: captured_build.update(kwargs),
    )
    monkeypatch.setattr(runner, "build_batch_inventory", lambda *_: object())
    monkeypatch.setattr(runner, "validate_local_batch", lambda *_: None)
    monkeypatch.setattr(runner, "commit_local_batch", lambda *_: None)
    monkeypatch.setattr(runner, "cleanup_ephemeral_batch", lambda *_args, **_kwargs: None)

    state = _RunnerState()
    context = SimpleNamespace(active_root=tmp_path / "active", artifacts_root=tmp_path / "artifacts")
    artifacts = runner.BatchArtifacts(
        frames_table=pd.DataFrame(),
        frame_native_tables={},
        child_tables={},
        visual_vectors=np.empty((0, 2), dtype=np.float32),
        visual_mapping=pd.DataFrame(),
        context_vectors=np.empty((0, 2), dtype=np.float32),
        context_mapping=pd.DataFrame(),
    )
    runner._process_one_batch(
        context,
        state,
        "L01",
        tmp_path / "archive",
        "L01-batch000",
        [_VIDEO_ID],
        lambda *_: artifacts,
        lambda _: bundle,
        dataset_version="batch-v1",
        visual_model_name="test/visual-encoder",
        visual_model_revision="visual-revision",
        context_model_name="test/evidence-encoder",
        context_model_revision="evidence-revision",
    )
    assert captured_build["visual_model_revision"] == "visual-revision"
    assert captured_build["context_model_revision"] == "evidence-revision"


def test_process_command_threads_encoder_revisions_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process command forwards both configured revisions to the runner."""

    entry = SimpleNamespace(archive_id="L01")
    plan = SimpleNamespace(entries=(entry,))
    window = SimpleNamespace(select=lambda _: (entry,))
    state_store = SimpleNamespace(create_or_resume_run=lambda *_args, **_kwargs: None)
    context = SimpleNamespace(run_root=tmp_path / "run")
    visual_encoder = EncoderConfig(
        backend="siglip",
        model_name="test/visual-encoder",
        revision="visual-revision",
    )
    evidence_encoder = EncoderConfig(
        backend="bge_m3",
        model_name="test/evidence-encoder",
        revision="evidence-revision",
    )
    encoders = {
        "visual_embedding": visual_encoder,
        "evidence_embedding": evidence_encoder,
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(prepare_pipeline.CustomPipelineConfig, "from_yaml", lambda _: object())
    monkeypatch.setattr(prepare_pipeline, "_build_runner_context", lambda *_: context)
    monkeypatch.setattr(prepare_pipeline.ArchivePlan, "from_urls", lambda _: plan)
    monkeypatch.setattr(prepare_pipeline, "ArchiveWorkWindow", lambda **_: window)
    monkeypatch.setattr(prepare_pipeline, "PipelineStateStore", lambda _: state_store)
    monkeypatch.setattr(prepare_pipeline, "_build_run_identity", lambda *_: object())
    monkeypatch.setattr(prepare_pipeline, "_make_produce_batch_artifacts", lambda *_: object())
    monkeypatch.setattr(prepare_pipeline, "_make_asr_bundle_factory", lambda *_: object())
    monkeypatch.setattr(
        prepare_pipeline,
        "_load_encoder_config",
        lambda _path, section: encoders[section],
    )
    monkeypatch.setattr(
        prepare_pipeline,
        "process_archive",
        lambda *_args, **kwargs: captured.update(kwargs) or ["L01-batch000"],
    )

    result = prepare_pipeline._cmd_process_archive(
        SimpleNamespace(
            config=tmp_path / "prepare.yaml",
            archive_urls=["https://example.test/L01.zip"],
            offset=0,
            limit=None,
            allow_offset_gap=False,
            version="batch-v1",
            batch_offset=0,
            batch_limit=None,
        )
    )

    assert result["committed_batches"] == {"L01": ["L01-batch000"]}
    assert captured["visual_model_revision"] == "visual-revision"
    assert captured["context_model_revision"] == "evidence-revision"


class _RunnerState:
    """Minimal state-store spy for the runner's batch-boundary unit test."""

    def get_batch(self, batch_id: str) -> None:
        """Report that the synthetic batch is not already committed."""

        return None

    def ensure_batch(self, batch_id: str, archive_id: str, video_ids: list[str]) -> None:
        """Accept creation of the synthetic batch."""

    def ensure_video(self, video_id: str, batch_id: str) -> None:
        """Accept creation of the synthetic video state."""

    def advance_batch(self, batch_id: str, stage: object) -> None:
        """Accept the runner's normal batch-stage transitions."""

    def advance_video(self, video_id: str, stage: object) -> None:
        """Accept the runner's normal video-stage transitions."""
