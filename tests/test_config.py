from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.enrichment.caption import CaptionJobConfig
from hcmai.llm.config import LLMServiceConfig


def test_baseline_config_matches_runtime_contract() -> None:
    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.dataset.frames_path.as_posix() == "data/metadata/frames.parquet"
    assert config.dataset.enrichment.caption_path == Path(
        "artifacts/enrichment/caption/frame_enrichment.parquet"
    )
    assert config.dataset.enrichment.ocr_path == Path(
        "artifacts/enrichment/ocr/frame_enrichment.parquet"
    )
    assert config.dataset.enrichment.asr_path == Path(
        "artifacts/enrichment/asr/frame_enrichment.parquet"
    )
    assert config.index.path.as_posix() == "artifacts/indexes/visual"
    assert config.index.caption_path.as_posix() == "artifacts/indexes/caption"
    assert config.search.candidate_count == 500
    assert config.search.rerank_count == 100
    assert config.search.temporal_window_ms == 3000
    assert config.inference.base_url == "https://api.iamphuckhang.dev"
    assert config.inference.local_embedding_fallback is True


def test_llm_config_is_the_model_authority() -> None:
    config = LLMServiceConfig.from_yaml("llm/config.yaml")

    assert (
        config.visual_embedding.model_name
        == "google/siglip2-base-patch16-224"
    )
    assert (
        config.caption_embedding.model_name
        == "BAAI/bge-m3"
    )
    assert config.visual_embedding.backend == "siglip"
    assert config.caption_embedding.backend == "bge_m3"
    assert config.reranker.checkpoint == "Qwen/Qwen3-VL-Reranker-2B"


def test_enrichment_config_is_loaded_from_root_yaml() -> None:
    config = CaptionJobConfig.from_yaml()

    assert config.caption.model_checkpoint == "microsoft/Florence-2-base-ft"
    assert config.caption.decoding["num_beams"] == 3
    assert config.caption.dataset_version == "hcmai2026_v1"
    assert config.frames_path == Path.cwd() / "data/metadata/frames.parquet"
    assert config.output_dir == Path.cwd() / "artifacts/enrichment/caption"
