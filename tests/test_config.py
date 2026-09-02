from pathlib import Path

import pytest
import yaml
from hcmai.common.config import (
    AppConfig,
    FusionConfig,
    LEGACY_DATASET_ROOT,
    REPOSITORY_ROOT,
    SearchConfig,
    TranscriptJobConfig,
    resolve_dataset_root,
    resolve_repository_path,
)
from hcmai.retrieval.models import RetrievalSource
from offline.enrichment.caption.config import CaptionJobConfig
from thundercompute.config import LLMServiceConfig


def test_baseline_config_matches_runtime_contract() -> None:
    config = AppConfig.from_yaml("configs/baseline.yaml")

    assert config.dataset.root == Path("data")
    assert config.dataset.frames_path == Path("artifacts/frame_store/frames.parquet")
    assert config.dataset.enrichment.caption_path == Path("artifacts/corpus/caption.parquet")
    assert config.dataset.enrichment.ocr_path == Path("artifacts/corpus/ocr_frames.parquet")
    assert config.dataset.enrichment.object_path == Path("artifacts/corpus/object_frames.parquet")
    assert config.dataset.enrichment.context_path == Path("artifacts/corpus/context.parquet")
    assert config.dataset.media_info_path == Path("data/media-info")
    assert config.dataset.enrichment.asr_path == Path(
        "artifacts/enrichment/asr/frame_enrichment.parquet"
    )
    assert config.index.path.as_posix() == "artifacts/indexes/visual"
    assert config.index.context_path.as_posix() == "artifacts/indexes/context"
    assert config.index.asr_segment_path.as_posix() == "artifacts/indexes/asr_segments"
    assert config.index.asr_projection_max_gap_ms == 5_000
    assert config.index.caption_path.as_posix() == "artifacts/indexes/caption"
    assert config.index.ocr_path.as_posix() == "artifacts/indexes/ocr"
    assert not hasattr(config.index, "asr_path")
    assert config.index.text_embedding_filenames == {
        RetrievalSource.CAPTION: "caption_embeddings.npy",
        RetrievalSource.OCR: "ocr_embeddings.npy",
        RetrievalSource.ASR: "asr_embeddings.npy",
    }
    assert config.search.fusion.source_weights == {source: 1.0 for source in RetrievalSource}
    assert config.search.alignment.lambda_gap == pytest.approx(1e-5)
    assert config.search.alignment.event_power == 1.0
    assert config.search.alignment.chunk_size == 65_536
    assert config.search.alignment.cluster_delta == 0.0
    assert config.inference.enabled is True
    assert config.inference.base_url == "https://api.iamphuckhang.dev"
def test_search_config_rejects_retired_progressive_options() -> None:
    """Fail configuration loading instead of silently reviving removed state."""

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SearchConfig.model_validate({"progressive": {}})


def test_fusion_weights_require_every_retrieval_source() -> None:
    """Do not allow an active source to inherit an implicit fusion weight."""

    with pytest.raises(ValueError, match="source_weights must configure"):
        FusionConfig(source_weights={RetrievalSource.VISUAL: 1.0})


def test_runtime_repository_paths_do_not_depend_on_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give data/ and artifacts/ one meaning for every backend launcher."""

    monkeypatch.chdir(tmp_path)

    assert resolve_repository_path("data") == REPOSITORY_ROOT / "data"
    assert resolve_repository_path("artifacts/frame_store/frames.parquet") == (
        REPOSITORY_ROOT / "artifacts/frame_store/frames.parquet"
    )


def test_legacy_frame_store_root_migrates_to_canonical_data_root() -> None:
    """Keep stale launch environments from looking for images in artifacts."""

    assert resolve_dataset_root("artifacts/frame_store") == REPOSITORY_ROOT / "data"
    assert resolve_dataset_root(LEGACY_DATASET_ROOT) == REPOSITORY_ROOT / "data"
    assert resolve_dataset_root("data") == REPOSITORY_ROOT / "data"


def test_llm_config_is_the_model_authority() -> None:
    config = LLMServiceConfig.from_yaml("thundercompute/config.yaml")

    assert config.visual_embedding.model_name == "google/siglip2-base-patch16-224"
    assert config.caption_embedding.model_name == "BAAI/bge-m3"
    assert config.visual_embedding.backend == "siglip"
    assert config.caption_embedding.backend == "bge_m3"
    assert config.visual_embedding.revision == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    assert config.caption_embedding.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert config.reranker.checkpoint == "Qwen/Qwen3-VL-Reranker-2B"
    assert config.query_preparation.model_checkpoint == "Qwen/Qwen3-4B"
    assert config.query_preparation.revision == "1cfa9a7208912126459214e8b04321603b3df60c"



def test_enrichment_config_is_loaded_from_root_yaml() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = {
        "version": "dataset_v1",
        "source": "custom_raw_video",
        "data_root": "runs/dataset_v1",
        "frame_store_id": "dataset_1",
        "frames_path": "artifacts/dataset_v1/frame_store/frames.parquet",
        "frame_store_output": "artifacts/dataset_v1/frame_store",
    }

    config = CaptionJobConfig.from_yaml(dataset=dataset)

    assert config.caption.model_checkpoint == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.caption.revision == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert config.caption.prompt == "qwen vl"
    assert config.caption.decoding["max_new_tokens"] == 96
    assert config.caption.dataset_version == "dataset_v1"
    assert config.dataset_root == project_root / "runs/dataset_v1"
    assert config.frames_path == project_root / "artifacts/dataset_v1/frame_store/frames.parquet"
    assert config.output_dir == project_root / "artifacts/enrichment/captions"
    assert config.frame_store_id == "dataset_1"

    transcript = TranscriptJobConfig.from_yaml("configs/prepare.yaml", dataset=dataset)
    assert transcript.asr.revision == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert transcript.diarization.revision == "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
    assert (
        transcript.frames_path == project_root / "artifacts/dataset_v1/frame_store/frames.parquet"
    )
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
        yaml.safe_dump({
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
            },},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root if working_directory == "repository" else tmp_path)

    config = CaptionJobConfig.from_yaml(config_path)

    assert config.dataset_root == project_root / "fixture/data"
    assert config.frames_path == project_root / "fixture/data/frames.parquet"
    assert config.output_dir == project_root / "fixture/captions"


def test_caption_job_rejects_non_path_dataset_root(tmp_path: Path) -> None:
    """Reject malformed YAML paths before passing them to ``pathlib``."""

    raw = yaml.safe_load(Path("configs/prepare.yaml").read_text(encoding="utf-8"))["enrichment"]
    raw["dataset"] = {
        "version": "dataset_v1",
        "data_root": 123,
        "frames_path": "artifacts/dataset_v1/frame_store/frames.parquet",
    }
    config_path = tmp_path / "enrichment.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="data_root"):
        CaptionJobConfig.from_yaml(config_path)
