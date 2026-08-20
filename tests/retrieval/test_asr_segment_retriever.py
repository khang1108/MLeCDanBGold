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
from hcmai.data.enrichment.transcripts.store import TranscriptStore
from hcmai.llm.config import LLMServiceConfig
from hcmai.retrieval.retriever.artifacts import fingerprint_files


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
                [2.0, 0.0] if "hello" in text.lower() else [0.0, 3.0]
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
    from hcmai.retrieval.retriever.segment.artifacts import build_segment_corpus

    _write_transcripts(tmp_path / "transcripts")
    store = TranscriptStore(tmp_path / "transcripts")

    texts, mapping = build_segment_corpus(store)

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
    from hcmai.retrieval.retriever.segment.artifacts import build_segment_corpus

    path = tmp_path / "failed.parquet"
    pd.DataFrame(
        [_segment_row("v1:0", 0, "failed", status=ProcessingStatus.FAILED)]
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no usable completed segments"):
        build_segment_corpus(TranscriptStore(path))


def test_asr_segment_artifact_builder_uses_evidence_encoder_and_lineage(
    tmp_path: Path,
) -> None:
    """Publish a loadable segment bundle fingerprinted from shards and manifests."""

    pytest.importorskip("faiss")
    pytest.importorskip("pyarrow")
    from hcmai.retrieval.retriever.pipeline import RetrievalService
    from hcmai.retrieval.retriever.segment.index import SegmentDenseIndex

    transcripts = tmp_path / "transcripts"
    parquet, manifest = _write_transcripts(transcripts)
    output = tmp_path / "asr-segments"
    pipeline_config, model_config = _build_configs(tmp_path, transcripts, output)
    encoder = FakeBGE()

    index = RetrievalService.build_asr_segment_artifacts(
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
    from hcmai.retrieval.retriever.segment.artifacts import transcript_lineage_files

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

    from hcmai.retrieval.retriever.segment.artifacts import build_asr_segment_artifacts

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

    from hcmai.llm.pipeline import LLMService
    from hcmai.retrieval.retriever.segment import artifacts

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
    from hcmai.retrieval.retriever.segment.artifacts import build_asr_segment_index

    _write_transcripts(tmp_path / "transcripts")

    with pytest.raises(ValueError, match="plain .npy filename"):
        build_asr_segment_index(
            TranscriptStore(tmp_path / "transcripts"),
            FakeBGE(),
            tmp_path / "index",
            embeddings_filename="../asr.npy",
            dataset_version="test-v1",
        )
