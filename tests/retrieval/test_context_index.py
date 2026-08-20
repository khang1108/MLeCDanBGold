"""Regression coverage for the dedicated context and ASR-segment index profile."""

from __future__ import annotations

from pathlib import Path

import pytest

from hcmai.common.config import (
    EncoderConfig,
    EnrichmentArtifactsConfig,
    FusionConfig,
    IndexConfig,
)
from hcmai.common.schemas import RetrievalSource, TaskType
from hcmai.common.utils.io import read_yaml
from hcmai.llm.config import LLMServiceConfig


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
    """Every task receives an explicit neutral context fusion weight by default."""

    config = FusionConfig()

    assert RetrievalSource.CONTEXT in config.task_weights[TaskType.KIS]
    assert set(config.task_weights[TaskType.KIS]) == set(RetrievalSource)


def test_context_and_segment_paths_are_dedicated_to_the_new_profile() -> None:
    """Context and transcript indexes do not overload legacy text-index fields."""

    enrichment = EnrichmentArtifactsConfig()
    index = IndexConfig()

    assert enrichment.context_path == Path(
        "artifacts/enrichment/context/frame_context_v1.parquet"
    )
    assert enrichment.transcripts_path == Path("artifacts/enrichment/transcripts")
    assert index.profile == "context_asr_segment"
    assert index.context_path == Path("artifacts/indexes/context")
    assert index.asr_segment_path == Path("artifacts/indexes/asr_segments")
    assert index.context_embedding_filename == "context_embeddings.npy"
    assert index.asr_segment_embedding_filename == "asr_embeddings.npy"
    assert index.asr_projection_max_gap_ms == 5_000


def test_legacy_text_embedding_filenames_do_not_absorb_context() -> None:
    """Context uses its own index artifact instead of changing legacy validation."""

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

    config = LLMServiceConfig.from_yaml("configs/indexing.models.yaml")

    assert config.visual_embedding.revision == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert (
        config.resolved_evidence_embedding.revision
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )


def test_indexing_config_uses_portable_corpus_paths_and_expected_counts() -> None:
    """The offline indexing plan remains portable between local and VM runs."""

    config = read_yaml("configs/indexing.yaml")

    assert config["dataset"]["expected_video_count"] == 873
    assert config["dataset"]["expected_frame_count"] == 177_321
    assert config["dataset"]["context_path"] == (
        "artifacts/enrichment/context/frame_context_v1.parquet"
    )
    assert config["indexes"] == {
        "visual": "artifacts/indexes/visual",
        "context": "artifacts/indexes/context",
        "asr_segments": "artifacts/indexes/asr_segments",
    }
    assert config["projection"]["max_projection_gap_ms"] == 5_000


def test_baseline_defers_context_profile_paths_until_profile_aware_startup() -> None:
    """Legacy startup remains on its legacy index paths until Task 10 owns loading."""

    baseline = read_yaml("configs/baseline.yaml")
    index = baseline["index"]

    assert "profile" not in index
    assert "context_path" not in index
    assert "asr_segment_path" not in index
    assert "context_embedding_filename" not in index
    assert "asr_segment_embedding_filename" not in index
    assert "asr_projection_max_gap_ms" not in index
    assert all("context" in weights for weights in baseline["search"]["fusion"]["task_weights"].values())
