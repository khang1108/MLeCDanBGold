"""Regression coverage for offline ASR segment corpus and index artifacts.

These tests keep Task 7 segment-native: frame projection and online retrieval
belong to the subsequent migration task.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import AppConfig, EncoderConfig, InferenceConfig
from hcmai.common.schemas import ProcessingStatus
from hcmai.common.utils.io import write_json, write_yaml
from hcmai.corpus.stores.transcript import TranscriptStore
from offline.enrichment.transcripts.artifacts import (
    load_transcript_artifact_records,
)
from thundercompute.config import LLMServiceConfig
from hcmai.retrieval.retriever.artifacts import fingerprint_files


def _canonical_frame(
    frame_id: str,
    timestamp_ms: int,
    frame_idx: int,
    *,
    video_id: str = "v1",
) -> dict[str, object]:
    """Return one canonical frame mapping row for online retrieval tests."""

    return {
        "frame_id": frame_id,
        "video_id": video_id,
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
        "image_path": f"/frames/{frame_id}.jpg",
        "width": 640,
        "height": 360,
    }


class FakeBGE:
    """Provide deterministic CPU-only BGE-shaped vectors for artifact tests."""

    config = EncoderConfig(
        backend="bge_m3",
        model_name="fake/bge-m3",
        batch_size=2,
        revision="a" * 40,
    )
    embedding_dim = 2
    resolved_revision = "b" * 40

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_text(self, texts, stats=None) -> np.ndarray:
        """Map fixture speech to small non-normalized vectors."""

        self.calls.append(list(texts))
        return np.asarray(
            [
                [1.0, 0.0] if "hello" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def _segment_row(
    segment_id: str,
    segment_index: int,
    text: str,
    *,
    status: ProcessingStatus = ProcessingStatus.COMPLETED,
) -> dict[str, object]:
    """Return one valid transcript row with explicit ASR provenance."""

    row: dict[str, object] = {
        "segment_id": segment_id,
        "video_id": "v1",
        "segment_index": segment_index,
        "start_ms": 1_000 + segment_index * 1_000,
        "end_ms": 2_000 + segment_index * 1_000,
        "text": text,
        "language": "en",
        "speaker_id": "speaker-1",
        "confidence": None if segment_index == 0 else 0.75,
        "status": status.value,
        "model_name": "test/asr",
        "model_revision": "c" * 40,
        "artifact_version": "asr-segment-v1",
        "error_code": None,
        "error_message": None,
    }
    if status is ProcessingStatus.FAILED:
        row["error_code"] = "provider_error"
        row["error_message"] = "failed"
    return row


def _write_transcripts(root: Path) -> tuple[Path, Path]:
    """Write one transcript shard plus its adjacent lineage manifest."""

    shard_dir = root / "L01"
    shard_dir.mkdir(parents=True)
    parquet = shard_dir / "v1.parquet"
    manifest = shard_dir / "v1.manifest.json"
    pd.DataFrame(
        [
            _segment_row("v1:0", 0, "  hello   world  "),
            _segment_row(
                "v1:1", 1, "do not embed", status=ProcessingStatus.FAILED
            ),
            _segment_row("v1:2", 2, "night market"),
        ]
    ).to_parquet(parquet, index=False)
    write_json({"video_id": "v1", "status": "completed"}, manifest)
    return parquet, manifest


def _build_configs(
    tmp_path: Path, transcripts: Path, output: Path
) -> tuple[Path, Path]:
    """Write pipeline and model configs for a segment artifact build."""

    pipeline_config = tmp_path / "baseline.yaml"
    model_config = tmp_path / "models.yaml"
    write_yaml(
        {
            "dataset": {
                "version": "test-v1",
                "enrichment": {"transcripts_path": str(transcripts)},
            },
            "index": {
                "asr_segment_path": str(output),
                "asr_segment_embedding_filename": "asr_embeddings.npy",
            },
        },
        pipeline_config,
    )
    write_yaml(
        {
            "caption_embedding": {
                "backend": "bge_m3",
                "name": "legacy/caption-bge",
            },
            "evidence_embedding": {
                "backend": "bge_m3",
                "name": "fake/bge-m3",
                "revision": "a" * 40,
            },
        },
        model_config,
    )
    return pipeline_config, model_config


def test_asr_segment_corpus_preserves_timeline_identity_and_provenance(
    tmp_path: Path,
) -> None:
    """Embed only completed normalized speech without inventing frame identity."""

    pytest.importorskip("pyarrow")
    from offline.indexes.asr_segment import build_segment_corpus

    _write_transcripts(tmp_path / "transcripts")
    records = load_transcript_artifact_records(tmp_path / "transcripts")

    texts, mapping = build_segment_corpus(records)

    assert texts == ["hello world", "night market"]
    assert mapping["embedding_index"].tolist() == [0, 1]
    assert mapping["segment_id"].tolist() == ["v1:0", "v1:2"]
    assert mapping["segment_index"].tolist() == [0, 2]
    assert mapping["start_ms"].tolist() == [1_000, 3_000]
    assert mapping["end_ms"].tolist() == [2_000, 4_000]
    assert mapping["language"].tolist() == ["en", "en"]
    assert mapping["speaker_id"].tolist() == ["speaker-1", "speaker-1"]
    assert mapping["confidence"].tolist()[0] is None
    assert mapping["model_name"].tolist() == ["test/asr", "test/asr"]
    assert mapping["model_revision"].tolist() == ["c" * 40, "c" * 40]
    assert mapping["artifact_version"].tolist() == [
        "asr-segment-v1",
        "asr-segment-v1",
    ]
    assert "frame_id" not in mapping.columns


def test_asr_segment_corpus_rejects_artifact_without_usable_segments(
    tmp_path: Path,
) -> None:
    """Fail an offline build when every transcript segment is incomplete."""

    pytest.importorskip("pyarrow")
    from offline.indexes.asr_segment import build_segment_corpus

    path = tmp_path / "failed.parquet"
    pd.DataFrame(
        [_segment_row("v1:0", 0, "failed", status=ProcessingStatus.FAILED)]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no usable completed segments"):
        build_segment_corpus(load_transcript_artifact_records(path))


def test_asr_segment_artifact_builder_uses_evidence_encoder_and_lineage(
    tmp_path: Path,
) -> None:
    """Publish a loadable segment bundle fingerprinted from shards and manifests."""

    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    from offline.indexes.asr_segment import build_asr_segment_artifacts
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex
    from hcmai.retrieval.retriever.segment.index import (
        CHECKSUM_FILENAMES,
        REQUIRED_INDEX_FILENAMES,
    )

    transcripts = tmp_path / "transcripts"
    parquet, manifest = _write_transcripts(transcripts)
    output = tmp_path / "asr-segments"
    pipeline_config, model_config = _build_configs(tmp_path, transcripts, output)
    encoder = FakeBGE()

    index = build_asr_segment_artifacts(
        pipeline_config, model_config, encoder=encoder
    )
    loaded = SegmentDenseIndex.load(output)

    assert encoder.calls == [["hello world", "night market"]]
    assert index.metadata.model_name == "fake/bge-m3"
    assert index.metadata.model_revision == "b" * 40
    assert index.metadata.source_fingerprint == fingerprint_files([parquet, manifest])
    assert loaded.metadata.source_fingerprint == index.metadata.source_fingerprint
    assert loaded.metadata.entity_kind == "segment"
    assert loaded.metadata.retrieval_source == "asr"
    assert {path.name for path in output.iterdir()} == {
        *REQUIRED_INDEX_FILENAMES,
        "asr_embeddings.npy",
    }
    assert set(loaded.metadata.checksums) == set(CHECKSUM_FILENAMES)
    assert loaded.metadata.schema_version == "dense-index-v2"
    assert loaded.mapping["segment_id"].tolist() == ["v1:0", "v1:2"]
    assert "frame_id" not in loaded.mapping.columns
    np.testing.assert_allclose(
        np.load(output / "asr_embeddings.npy"), loaded.vectors
    )


def test_asr_segment_lineage_fingerprints_every_shard_and_optional_manifest(
    tmp_path: Path,
) -> None:
    """Include every Parquet shard and only present adjacent manifests."""

    pytest.importorskip("pyarrow")
    from offline.indexes.asr_segment import transcript_lineage_files

    root = tmp_path / "transcripts"
    first, manifest = _write_transcripts(root)
    second = root / "L02" / "v2.parquet"
    second.parent.mkdir(parents=True)
    pd.DataFrame([_segment_row("v2:0", 0, "second shard")]).assign(
        video_id="v2"
    ).to_parquet(second, index=False)

    assert transcript_lineage_files(root) == (first, manifest, second)


@pytest.mark.parametrize("transcript_kind", ["missing", "empty_directory"])
def test_asr_segment_artifact_builder_rejects_missing_transcript_data(
    tmp_path: Path, transcript_kind: str
) -> None:
    """Require at least one non-empty transcript Parquet before model loading."""

    from offline.indexes.asr_segment import build_asr_segment_artifacts

    transcripts = tmp_path / "transcripts"
    if transcript_kind == "empty_directory":
        transcripts.mkdir()
    output = tmp_path / "asr-segments"
    pipeline_config, model_config = _build_configs(tmp_path, transcripts, output)

    with pytest.raises(FileNotFoundError, match="Transcript artifact"):
        build_asr_segment_artifacts(
            pipeline_config, model_config, encoder=FakeBGE()
        )


def test_remote_asr_segment_encoder_uses_text_source_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted segment embeddings select the generic BGE text endpoint."""

    from thundercompute.pipeline import LLMService
    from offline.indexes import asr_segment as artifacts

    captured: dict[str, object] = {}
    remote_encoder = FakeBGE()
    settings = AppConfig(inference=InferenceConfig(enabled=True))
    models = LLMServiceConfig(evidence_embedding=remote_encoder.config)

    monkeypatch.setattr(LLMService, "remote", lambda *args: object())

    def create_remote_adapter(client, config, embedding_dim, source):
        captured.update(
            client=client,
            config=config,
            embedding_dim=embedding_dim,
            source=source,
        )
        return remote_encoder

    monkeypatch.setattr(
        artifacts.EmbeddingService,
        "create_remote_adapter",
        create_remote_adapter,
    )

    assert artifacts._segment_encoder(settings, models, None) is remote_encoder
    assert captured["source"] == "text"


def test_asr_segment_builder_rejects_pathlike_embedding_filename(
    tmp_path: Path,
) -> None:
    """Keep supplemental vector publication inside the index directory."""

    pytest.importorskip("pyarrow")
    from offline.indexes.asr_segment import build_asr_segment_index

    _write_transcripts(tmp_path / "transcripts")

    with pytest.raises(ValueError, match="plain .npy filename"):
        build_asr_segment_index(
            TranscriptStore(tmp_path / "transcripts"),
            FakeBGE(),
            tmp_path / "index",
            embeddings_filename="../asr.npy",
            dataset_version="test-v1",
        )


def _online_retrieval_fixture(tmp_path: Path):
    """Build a tiny segment index and canonical frame store for online tests."""

    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    from hcmai.corpus.stores.frame import FrameStore
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            _canonical_frame("f1", 1_500, 15),
            _canonical_frame("f2", 5_000, 50),
        ]
    ).to_parquet(frames_path, index=False)
    mapping = pd.DataFrame(
        [
            {
                "embedding_index": 0,
                "segment_id": "v1:strong",
                "video_id": "v1",
                "segment_index": 0,
                "start_ms": 1_000,
                "end_ms": 2_000,
            },
            {
                "embedding_index": 1,
                "segment_id": "v1:weak",
                "video_id": "v1",
                "segment_index": 1,
                "start_ms": 1_200,
                "end_ms": 1_800,
            },
            {
                "embedding_index": 2,
                "segment_id": "v1:second",
                "video_id": "v1",
                "segment_index": 2,
                "start_ms": 4_500,
                "end_ms": 5_500,
            },
        ]
    )
    vectors = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32
    )
    index = SegmentDenseIndex.build(
        vectors,
        mapping,
        dataset_version="test-v1",
        model_name="fake/bge-m3",
    )
    return FrameStore(frames_path), index


def test_asr_segment_retriever_projects_deduplicates_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    """Emit ranked canonical frames while retaining the strongest ASR segment."""

    from hcmai.common.schemas import RetrievalSource
    from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever

    frames, index = _online_retrieval_fixture(tmp_path)
    retriever = ASRSegmentRetriever(
        FakeBGE(), index, frames, max_projection_gap_ms=1_000
    )

    result = retriever.search("hello", top_k=3)

    assert [candidate.frame_id for candidate in result] == ["f1", "f2"]
    assert result[0].source_scores == {RetrievalSource.ASR: 1.0}
    assert result[0].source_ranks == {RetrievalSource.ASR: 1}
    assert result[1].source_ranks == {RetrievalSource.ASR: 2}
    assert result[0].metadata == {
        "frame": {
            "frame_id": "f1",
            "video_id": "v1",
            "frame_idx": 15,
            "timestamp_ms": 1_500,
        },
        "asr_segment": {
            "segment_id": "v1:strong",
            "start_ms": 1_000,
            "end_ms": 2_000,
            "projection_kind": "inside_segment",
            "projection_distance_ms": 0,
            "segment_score": 1.0,
        },
    }


def test_asr_segment_retriever_rejects_incompatible_query_batch(
    tmp_path: Path,
) -> None:
    """Enforce the same model, dimension, family, and normalization contract."""

    from dataclasses import replace

    from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever

    frames, index = _online_retrieval_fixture(tmp_path)
    retriever = ASRSegmentRetriever(FakeBGE(), index, frames)
    batch = retriever.encode(["hello"])

    wrong_family = replace(
        batch,
        embeddings=(
            replace(
                batch.embeddings[0],
                query=replace(batch.embeddings[0].query, source_family="visual"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="source family"):
        retriever.search_vectors(wrong_family)


def test_context_and_asr_segment_fusion_encode_text_batch_once(
    tmp_path: Path,
) -> None:
    """Reuse one BGE text-family batch across Context and projected ASR indexes."""

    from hcmai.common.config import FusionConfig
    from hcmai.common.schemas import RetrievalSource
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from hcmai.retrieval.retriever.fusion import RRFFusionRetriever
    from hcmai.retrieval.retriever.segment.retriever import ASRSegmentRetriever
    from hcmai.retrieval.retriever.text.retriever import ContextRetriever

    frames, segment_index = _online_retrieval_fixture(tmp_path)
    context_index = DenseIndex.build(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        pd.DataFrame(
            [
                {
                    "embedding_index": 0,
                    "frame_id": "f1",
                    "video_id": "v1",
                    "frame_idx": 15,
                    "timestamp_ms": 1_500,
                }
            ]
        ),
        dataset_version="test-v1",
        model_name="fake/bge-m3",
        show_progress=False,
    )
    encoder = FakeBGE()
    fusion = RRFFusionRetriever(
        [
            ContextRetriever(encoder, context_index),
            ASRSegmentRetriever(encoder, segment_index, frames),
        ],
        FusionConfig(
            required_sources={RetrievalSource.CONTEXT, RetrievalSource.ASR}
        ),
    )

    results = fusion.search_batch(["hello", "night market"], top_k=3)

    assert len(results) == 2
    assert encoder.calls == [["hello", "night market"]]
    assert results[0][0].metadata["asr_segment"]["segment_id"] == "v1:strong"
