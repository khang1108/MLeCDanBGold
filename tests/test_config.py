from hcmai.common.config import AppConfig


def test_baseline_config_matches_runtime_contract() -> None:
    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.dataset.frames_path.as_posix() == "data/metadata/frames.parquet"
    assert config.index.path.as_posix() == "artifacts/indexes/visual"
    assert config.models.reranker.model_name == "Qwen/Qwen3-VL-Reranker-2B"
