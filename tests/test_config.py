from pathlib import Path

from hcmai.common.config import AppConfig
from hcmai.enrichment.caption import CaptionJobConfig


def test_baseline_config_matches_runtime_contract() -> None:
    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.dataset.frames_path.as_posix() == "data/metadata/frames.parquet"
    assert config.index.path.as_posix() == "artifacts/indexes/visual"
    assert config.models.reranker.model_name == "Qwen/Qwen3-VL-Reranker-2B"
    assert config.inference.base_url == "https://api.iamphuckhang.dev"
    assert config.inference.local_embedding_fallback is True


def test_enrichment_config_is_loaded_from_root_yaml() -> None:
    config = CaptionJobConfig.from_yaml()

    assert config.caption.model_checkpoint == "microsoft/Florence-2-base-ft"
    assert config.caption.decoding["num_beams"] == 3
    assert config.caption.dataset_version == "hcmai2026_v1"
    assert config.frames_path == Path.cwd() / "data/metadata/frames.parquet"
    assert config.output_dir == Path.cwd() / "artifacts/enrichment/caption"
