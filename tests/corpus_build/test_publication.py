"""Tests for artifact inventory and group publication logic."""

from pathlib import Path

from offline.ingestion.corpus_build.config import S3CorpusPreparationConfig
from offline.ingestion.corpus_build.pipeline import PreparationPaths
from offline.ingestion.corpus_build.publish import inventory_artifacts


def _setup_fixture(tmp_path: Path) -> tuple[Path, PreparationPaths, S3CorpusPreparationConfig]:
    work_root = tmp_path / "work"
    work_root.mkdir()
    config = S3CorpusPreparationConfig.model_validate({
        "corpus_revision": "test-v1",
        "work_root": str(work_root),
        "models": {
            "caption": {"model_name": "test", "revision": "0" * 40},
            "ocr": {"model_name": "test", "revision": "0" * 40},
            "asr": {"model_name": "test", "revision": "0" * 40},
            "diarization": {"model_name": "test", "revision": "0" * 40},
            "visual_embedding": {"model_name": "test", "revision": "0" * 40},
            "text_embedding": {"model_name": "test", "revision": "0" * 40},
        },
        "preprocessing": {
            "s3": {
                "bucket": "test-bucket",
                "videos_prefix": "videos",
                "artifacts_prefix": "test-artifacts",
                "smoke_artifacts_prefix": "smoke-test-artifacts",
                "staging_root": str(work_root / "staging"),
            },
        },
    })
    paths = PreparationPaths.from_config(config, None)
    
    # Create fake artifacts
    paths.artifacts_root.mkdir(parents=True, exist_ok=True)
    paths.frame_store_root.mkdir(parents=True, exist_ok=True)
    paths.visual_index_root.mkdir(parents=True, exist_ok=True)
    
    (paths.frame_store_root / "frames.parquet").write_text("frames")
    (paths.visual_index_root / "dense.index").write_text("index")
    
    return work_root, paths, config


def test_inventory_artifacts(tmp_path: Path) -> None:
    _, paths, _ = _setup_fixture(tmp_path)
    
    files = inventory_artifacts(paths.artifacts_root)
    assert len(files) == 2
    paths_list = [f.path for f in files]
    assert "frame_store/frames.parquet" in paths_list
    assert "indexes/visual/dense.index" in paths_list
    assert all(f.size > 0 for f in files)
