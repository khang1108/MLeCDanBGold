from pathlib import Path

import pytest
import yaml

from hcmai.common.config import AppConfig, TranscriptJobConfig
from hcmai.common.schemas import RetrievalSource, TaskType, VQABaselineProfile
from hcmai.data.enrichment.caption.config import CaptionJobConfig
from hcmai.llm.config import LLMServiceConfig


def test_baseline_config_matches_runtime_contract() -> None:
    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.dataset.root == Path("artifacts/frame_store")
    assert config.dataset.frames_path == Path("artifacts/frame_store/frames.parquet")
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
    assert config.index.ocr_path.as_posix() == "artifacts/indexes/ocr"
    assert config.index.asr_path.as_posix() == "artifacts/indexes/asr"
    assert config.index.text_embedding_filenames == {
        RetrievalSource.CAPTION: "caption_embeddings.npy",
        RetrievalSource.OCR: "ocr_embeddings.npy",
        RetrievalSource.ASR: "asr_embeddings.npy",
    }
    assert config.search.fusion.task_weights[TaskType.KIS] == {
        source: 1.0 for source in RetrievalSource
    }
    assert config.search.candidate_count == 500
    assert config.search.rerank_count == 100
    assert config.search.temporal_window_ms == 3000
    assert config.vqa.default_profile is VQABaselineProfile.LOCALIZER
    assert set(config.vqa.profiles) == set(VQABaselineProfile)
    assert config.vqa.profiles[VQABaselineProfile.SINGLE_FRAME].max_vlm_calls == 1
    assert config.inference.base_url == "https://api.iamphuckhang.dev"


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
    assert (
        config.visual_embedding.revision
        == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    )
    assert (
        config.caption_embedding.revision
        == "5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert config.reranker.checkpoint == "Qwen/Qwen3-VL-Reranker-2B"


def test_enrichment_config_is_loaded_from_root_yaml() -> None:
    config = CaptionJobConfig.from_yaml()
    project_root = Path(__file__).resolve().parents[1]

    assert (
        config.caption.model_checkpoint
        == "florence-community/Florence-2-base-ft"
    )
    assert config.caption.revision == "0b03b6f15a4a211370fb204aee4e7dd48887ea37"
    assert config.caption.decoding["num_beams"] == 3
    assert config.caption.dataset_version == "hcmai2026_v1"
    assert config.dataset_root == project_root / "artifacts/frame_store"
    assert config.frames_path == project_root / "artifacts/frame_store/frames.parquet"
    assert config.output_dir == project_root / "artifacts/enrichment/caption"

    transcript = TranscriptJobConfig.from_yaml("configs/enrichment.yaml")
    assert transcript.asr.revision == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert transcript.diarization.revision == "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
    assert transcript.frames_path == project_root / "artifacts/frame_store/frames.parquet"
    assert transcript.frame_enrichment_path == (
        project_root / "artifacts/enrichment/asr/frame_enrichment.parquet"
    )


@pytest.mark.parametrize("working_directory", ["repository", "external"])
def test_relative_caption_paths_do_not_depend_on_process_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    working_directory: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "enrichment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "version": "fixture-v1",
                    "root": "fixture/data",
                    "frames_path": "fixture/data/frames.parquet",
                },
                "caption": {
                    "name": "fixture/model",
                    "revision": "fixture-revision",
                    "prompt": "<CAPTION>",
                    "decoding": {},
                    "device": "cpu",
                    "precision": "fp32",
                    "dtype": "float32",
                    "image_size": 8,
                    "batch_size": 1,
                    "enrichment_version": "fixture-caption-v1",
                    "write_interval": 1,
                    "output_dir": "fixture/captions",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root if working_directory == "repository" else tmp_path)

    config = CaptionJobConfig.from_yaml(config_path)

    assert config.dataset_root == project_root / "fixture/data"
    assert config.frames_path == project_root / "fixture/data/frames.parquet"
    assert config.output_dir == project_root / "fixture/captions"
