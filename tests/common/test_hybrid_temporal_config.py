"""Tests for hybrid temporal evidence configuration."""

from pathlib import Path

import pytest
from hcmai.common.config import AppConfig, HybridTemporalConfig, IndexConfig, SearchConfig
from hcmai.retrieval.models import RetrievalSource
from pydantic import ValidationError


def test_hybrid_defaults() -> None:
    """Use neutral starting weights for the unmeasured baseline."""

    config = HybridTemporalConfig()

    assert config.dense.visual_weight == pytest.approx(1 / 3)
    assert config.dense.context_weight == pytest.approx(1 / 3)
    assert config.dense.asr_weight == pytest.approx(1 / 3)
    assert config.dense_weight == pytest.approx(0.5)
    assert config.bm25_weight == pytest.approx(0.5)


def test_dense_weights_must_sum_to_one() -> None:
    """Reject dense mixtures whose scale drifts with configuration."""

    with pytest.raises(ValidationError, match="dense temporal weights"):
        HybridTemporalConfig(
            dense={
                "visual_weight": 1,
                "context_weight": 1,
                "asr_weight": 1,
        })


def test_hybrid_weights_must_sum_to_one() -> None:
    """Reject hybrid mixtures that do not form a convex combination."""

    with pytest.raises(ValidationError, match="hybrid temporal weights"):
        HybridTemporalConfig(dense_weight=0.7, bm25_weight=0.7)


def test_baseline_exposes_bm25_artifact_path() -> None:
    """Keep the runtime BM25 location explicit and reproducible."""

    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.index.bm25_path.as_posix() == "artifacts/indexes/bm25"


def test_index_config_uses_segment_asr_without_frame_native_asr_path() -> None:
    """Keep complete-ASR projection configuration without a frame-Dense index."""

    config = IndexConfig()

    assert config.asr_segment_path == Path("artifacts/indexes/asr_segments")
    assert config.asr_projection_max_gap_ms == 5_000
    assert not hasattr(config, "asr_path")


def test_text_embedding_filenames_require_every_text_source() -> None:
    """Reject incomplete offline text-index filename mappings."""

    with pytest.raises(ValueError, match="must configure caption, ocr, and asr"):
        IndexConfig(
            text_embedding_filenames={
                RetrievalSource.CAPTION: "caption_embeddings.npy",
                RetrievalSource.OCR: "ocr_embeddings.npy",
            }
        )


@pytest.mark.parametrize("filename", ["../asr.npy", "nested/asr.npy", "asr.txt"])
def test_text_embedding_filenames_reject_unsafe_names(filename: str) -> None:
    """Keep configured artifacts within their selected index directory."""

    with pytest.raises(ValueError, match="plain .npy filenames"):
        IndexConfig(
            text_embedding_filenames={
                RetrievalSource.CAPTION: "caption_embeddings.npy",
                RetrievalSource.OCR: "ocr_embeddings.npy",
                RetrievalSource.ASR: filename,
            }
        )


def test_temporal_event_limit_is_conservative_and_configurable() -> None:
    """Default to 32 events while allowing stricter deployment limits."""

    assert SearchConfig().max_temporal_event_count == 32
    assert SearchConfig(max_temporal_event_count=8).max_temporal_event_count == 8

    with pytest.raises(ValueError):
        SearchConfig(max_temporal_event_count=33)
