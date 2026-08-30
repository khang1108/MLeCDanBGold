"""Regression coverage for the dedicated context and ASR-segment index profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import (
    AppConfig,
    EncoderConfig,
    EnrichmentArtifactsConfig,
    FusionConfig,
    InferenceConfig,
    IndexConfig,
)
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.io import read_yaml, write_json, write_yaml
from hcmai.corpus.stores import FrameContextStore, FrameStore
from thundercompute.config import LLMServiceConfig
from hcmai.retrieval.retriever.artifacts import fingerprint_files


class FakeBGE:
    """Provide deterministic CPU-only BGE-shaped vectors for context tests."""

    config = EncoderConfig(
        backend="bge_m3",
        model_name="fake/bge-m3",
        batch_size=2,
    )
    embedding_dim = 2

    def encode_text(self, texts, stats=None) -> np.ndarray:
        """Map fixture text to a small non-normalized embedding space."""

        return np.asarray(
            [
                [2.0, 0.0]
                if "cable car" in text.lower()
                else [0.0, 3.0]
                if "market" in text.lower()
                else [1.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def _context_data(
    tmp_path: Path,
    context_texts: list[tuple[str, str | None, str, int]],
) -> tuple[FrameStore, FrameContextStore]:
    """Write a hand-checkable canonical frame and FrameContext fixture."""

    pytest.importorskip("pyarrow")

    frames = tmp_path / "frames.parquet"
    context = tmp_path / "frame_context_v1.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "frame_idx": 10 + position,
                "keyframe_order": position + 1,
                "timestamp_ms": timestamp_ms,
                "image_path": f"{frame_id}.jpg",
                "width": 10,
                "height": 10,
            }
            for position, (frame_id, _, video_id, timestamp_ms) in enumerate(
                context_texts
            )
        ]
    ).to_parquet(frames, index=False)
    pd.DataFrame(
        [
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "frame_idx": 10 + position,
                "timestamp_ms": timestamp_ms,
                "caption_text": None,
                "ocr_text": None,
                "object_summary": None,
                "context_text": context_text,
                "caption_available": False,
                "ocr_quality": 0.0,
                "object_count": 0,
                "context_version": "frame-context-v1",
                "caption_version": "caption-v1",
                "ocr_version": "ocr-v1",
                "object_version": "objects-v1",
                "frame_store_id": None,
            }
            for position, (frame_id, context_text, video_id, timestamp_ms) in enumerate(
                context_texts
            )
        ]
    ).to_parquet(context, index=False)
    return FrameStore(frames), FrameContextStore(context)


@pytest.fixture
def context_stores(tmp_path: Path) -> tuple[FrameStore, FrameContextStore]:
    """Return one usable and one empty FrameContext record."""

    return _context_data(
        tmp_path,
        [
            ("f1", "[CAPTION]\nA red cable car.", "v1", 1000),
            ("f2", None, "v1", 2000),
        ],
    )


@pytest.fixture
def fake_bge() -> FakeBGE:
    """Supply a fake BGE encoder so tests never need a GPU."""

    return FakeBGE()


def _context_build_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a Context builder config with distinct legacy/evidence encoders."""

    pipeline_config = tmp_path / "baseline.yaml"
    model_config = tmp_path / "models.yaml"
    output = tmp_path / "context-index"
    write_yaml(
        {
            "dataset": {
                "version": "test-v1",
                "frames_path": str(tmp_path / "frames.parquet"),
                "enrichment": {
                    "context_path": str(tmp_path / "frame_context_v1.parquet")
                },
            },
            "index": {
                "context_path": str(output),
                "context_embedding_filename": "context_embeddings.npy",
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
            },
        },
        model_config,
    )
    return pipeline_config, model_config, output


def test_context_corpus_embeds_only_non_empty_context(
    context_stores: tuple[FrameStore, FrameContextStore],
) -> None:
    """Empty derived text is omitted rather than replaced with synthetic evidence."""

    pytest.importorskip("faiss")
    from offline.indexes.text import _context_corpus

    texts, mapping = _context_corpus(*context_stores)

    assert texts == ["[CAPTION]\nA red cable car."]
    assert mapping["frame_id"].tolist() == ["f1"]
    assert mapping["timestamp_ms"].tolist() == [1000]


@pytest.mark.parametrize(
    "source",
    [RetrievalSource.CONTEXT, RetrievalSource.ASR],
)
def test_text_encoding_uses_configured_batch_size_without_legacy_cap(
    source: RetrievalSource,
) -> None:
    """Context and ASR builds must pass the configured large BGE batch through."""

    from offline.indexes.text import _encode_texts

    class RecordingBGE:
        """Record input sizes while returning deterministic non-empty vectors."""

        config = EncoderConfig(
            backend="bge_m3", model_name="fake/bge-m3", batch_size=128
        )
        embedding_dim = 2

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def encode_text(self, texts, stats=None) -> np.ndarray:
            self.batch_sizes.append(len(texts))
            return np.tile(
                np.asarray([[0.0, 1.0]], dtype=np.float32),
                (len(texts), 1),
            )

    encoder = RecordingBGE()
    vectors = _encode_texts(["evidence"] * 130, encoder, source)

    assert vectors.shape == (130, 2)
    assert encoder.batch_sizes == [128, 2]


def test_context_index_is_frame_native_and_keeps_supplemental_vectors(
    fake_bge: FakeBGE,
    context_stores: tuple[FrameStore, FrameContextStore],
    tmp_path: Path,
) -> None:
    """Context publication retains frame identity and outer-bundle embeddings."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from hcmai.retrieval.retriever.dense.index import (
        CHECKSUM_FILENAMES,
        REQUIRED_INDEX_FILENAMES,
    )
    from offline.indexes.text import build_context_index

    output = tmp_path / "context-index"
    index = build_context_index(
        *context_stores,
        fake_bge,
        output,
        embeddings_filename="context_embeddings.npy",
        dataset_version="test-v1",
    )
    loaded = DenseIndex.load(output)

    assert index.mapping["frame_id"].tolist() == ["f1"]
    assert loaded.mapping["video_id"].tolist() == ["v1"]
    assert loaded.metadata.entity_kind == "frame"
    assert loaded.metadata.retrieval_source == "context"
    assert {path.name for path in output.iterdir()} == {
        *REQUIRED_INDEX_FILENAMES,
        "context_embeddings.npy",
    }
    assert set(loaded.metadata.checksums) == set(CHECKSUM_FILENAMES)
    assert loaded.metadata.schema_version == "dense-index-v2"
    assert (output / "context_embeddings.npy").is_file()
    np.testing.assert_allclose(
        np.load(output / "context_embeddings.npy"), loaded.vectors
    )


def test_context_artifact_builder_uses_evidence_encoder_and_manifest_lineage(
    fake_bge: FakeBGE,
    tmp_path: Path,
) -> None:
    """Configured Context builds fingerprint its source artifact and manifest."""

    pytest.importorskip("faiss")
    from hcmai.retrieval.retriever.dense.index import DenseIndex
    from offline.indexes.text import build_context_artifacts

    _context_data(
        tmp_path,
        [("f1", "[CAPTION]\nA red cable car.", "v1", 1000)],
    )
    frames = tmp_path / "frames.parquet"
    context = tmp_path / "frame_context_v1.parquet"
    manifest = tmp_path / "manifest.json"
    write_json(
        {
            "context_version": "frame-context-v1",
            "caption_version": "caption-v1",
            "ocr_version": "ocr-v1",
            "object_version": "objects-v1",
        },
        manifest,
    )
    pipeline_config, model_config, output = _context_build_configs(tmp_path)

    index = build_context_artifacts(
        pipeline_config, model_config, encoder=fake_bge
    )

    assert index.metadata.model_name == "fake/bge-m3"
    assert index.metadata.source_fingerprint == fingerprint_files([context, manifest])
    assert DenseIndex.load(output).metadata.retrieval_source == "context"


@pytest.mark.parametrize("manifest_contents", [None, ""], ids=["missing", "empty"])
def test_context_artifact_builder_requires_non_empty_adjacent_manifest(
    fake_bge: FakeBGE,
    tmp_path: Path,
    manifest_contents: str | None,
) -> None:
    """Context lineage cannot be built from a parquet file alone or an empty manifest."""

    from offline.indexes.text import build_context_artifacts

    _context_data(
        tmp_path,
        [("f1", "[CAPTION]\nA red cable car.", "v1", 1000)],
    )
    if manifest_contents is not None:
        (tmp_path / "manifest.json").write_text(manifest_contents)
    pipeline_config, model_config, _ = _context_build_configs(tmp_path)

    with pytest.raises(FileNotFoundError, match="CONTEXT manifest artifact"):
        build_context_artifacts(pipeline_config, model_config, encoder=fake_bge)


def test_remote_context_encoder_uses_text_source_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted Context embeddings use the BGE text endpoint, not ``context``."""

    from thundercompute.pipeline import LLMService
    from offline.indexes import text as artifacts

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

    assert artifacts._context_encoder(settings, models, None) is remote_encoder
    assert captured["source"] == "text"


def test_evidence_embedding_falls_back_to_caption_embedding() -> None:
    """Legacy model files continue to provide the generic evidence encoder."""

    config = LLMServiceConfig(
        caption_embedding=EncoderConfig(backend="bge_m3", model_name="BAAI/bge-m3")
    )

    assert config.resolved_evidence_embedding.model_name == "BAAI/bge-m3"


def test_explicit_evidence_embedding_wins_over_caption_embedding(
    tmp_path: Path,
) -> None:
    """The dedicated evidence block overrides the rollback caption setting."""

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
visual_embedding:
  backend: siglip
  model_name: visual/model
caption_embedding:
  backend: bge_m3
  model_name: legacy/caption
evidence_embedding:
  backend: bge_m3
  model_name: dedicated/evidence
""",
        encoding="utf-8",
    )

    config = LLMServiceConfig.from_yaml(config_path)

    assert config.visual_embedding.model_name == "visual/model"
    assert config.caption_embedding.model_name == "legacy/caption"
    assert config.resolved_evidence_embedding.model_name == "dedicated/evidence"


def test_fusion_accepts_context_as_a_source() -> None:
    """Context receives an explicit neutral fusion weight by default."""

    config = FusionConfig()

    assert RetrievalSource.CONTEXT in config.source_weights
    assert set(config.source_weights) == set(RetrievalSource)


def test_context_and_segment_paths_are_dedicated_to_the_runtime() -> None:
    """Context and transcript indexes do not overload text-index fields."""

    enrichment = EnrichmentArtifactsConfig()
    index = IndexConfig()

    assert enrichment.context_path == Path(
        "artifacts/enrichment/context/frame_context_v1.parquet"
    )
    assert enrichment.transcripts_path == Path("artifacts/enrichment/transcripts")
    assert index.context_path == Path("artifacts/indexes/context")
    assert index.asr_segment_path == Path("artifacts/indexes/asr_segments")
    assert index.context_embedding_filename == "context_embeddings.npy"
    assert index.asr_segment_embedding_filename == "asr_embeddings.npy"
    assert index.asr_projection_max_gap_ms == 5_000


def test_text_embedding_filenames_do_not_absorb_context() -> None:
    """Context uses its own index artifact and filename contract."""

    with pytest.raises(ValueError, match="caption, ocr, and asr"):
        IndexConfig(
            text_embedding_filenames={
                RetrievalSource.CAPTION: "caption_embeddings.npy",
                RetrievalSource.OCR: "ocr_embeddings.npy",
                RetrievalSource.ASR: "asr_embeddings.npy",
                RetrievalSource.CONTEXT: "context_embeddings.npy",
            }
        )


def test_pinned_indexing_model_config_has_an_explicit_evidence_encoder() -> None:
    """Offline index builds resolve exact visual and text model revisions."""

    config = LLMServiceConfig.from_yaml("configs/prepare.yaml", section="models")

    assert config.visual_embedding.revision == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert (
        config.resolved_evidence_embedding.revision
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_indexing_config_uses_portable_corpus_paths_and_expected_counts() -> None:
    """The offline YAML keeps policies while dataset inputs come from CLI."""

    config = read_yaml("configs/prepare.yaml")["indexing"]

    assert "dataset" not in config
    assert config["indexes"] == {
        "visual": "artifacts/indexes/visual",
        "context": "artifacts/indexes/context",
        "asr_segments": "artifacts/indexes/asr_segments",
    }
    assert config["projection"]["max_projection_gap_ms"] == 5_000


def test_baseline_enables_context_and_segment_startup() -> None:
    """Baseline explicitly configures the current context/segment path."""

    baseline = read_yaml("configs/baseline.yaml")
    index = baseline["index"]

    assert index["context_path"] == "artifacts/indexes/context"
    assert index["asr_segment_path"] == "artifacts/indexes/asr_segments"
    assert index["asr_projection_max_gap_ms"] == 5_000
    assert "context" in baseline["search"]["fusion"]["source_weights"]
