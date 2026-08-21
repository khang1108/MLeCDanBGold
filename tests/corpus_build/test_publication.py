"""Tests for the complete-bundle publication logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hcmai.data.corpus_build.config import S3CorpusPreparationConfig
from hcmai.data.corpus_build.pipeline import PreparationPaths
from hcmai.data.corpus_build.publish import inventory_artifacts, publish_run_artifacts


def _setup_fixture(tmp_path: Path) -> tuple[Path, PreparationPaths, S3CorpusPreparationConfig]:
    work_root = tmp_path / "work"
    work_root.mkdir()
    config = S3CorpusPreparationConfig.model_validate({
        "corpus_revision": "test-v1",
        "work_root": str(work_root),
        "models": {
            "dino": {"model_name": "test", "revision": "0" * 40},
            "caption": {"model_name": "test", "revision": "0" * 40},
            "ocr": {"model_name": "test", "revision": "0" * 40},
            "asr": {"model_name": "test", "revision": "0" * 40},
            "diarization": {"model_name": "test", "revision": "0" * 40},
            "visual_embedding": {"model_name": "test", "revision": "0" * 40},
            "text_embedding": {"model_name": "test", "revision": "0" * 40},
        },
        "preprocessing": {
            "output_root": str(work_root / "preprocessing"),
            "dino_model": "test",
            "dino_revision": "0" * 40,
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


def test_publish_run_artifacts_success(tmp_path: Path, monkeypatch) -> None:
    _, paths, config = _setup_fixture(tmp_path)
    
    mock_client = MagicMock()
    # Mock out _verify_remote_size for these tests to avoid size mismatch errors on JSON strings
    from hcmai.data.corpus_build import publish
    monkeypatch.setattr(publish, "_verify_remote_size", lambda *args, **kwargs: None)
    
    pub = publish_run_artifacts(mock_client, paths, config, limit=None)
    
    assert pub.bucket == "test-bucket"
    assert pub.file_count == 2
    assert pub.latest_key == "test-artifacts/latest.json"
    assert pub.version_prefix.startswith(f"test-artifacts/versions/{pub.bundle_id}")

    # Verify upload_file called for each artifact
    assert mock_client.upload_file.call_count == 2
    
    # Verify put_object called for _SUCCESS and latest
    assert mock_client.put_object.call_count == 2
    calls = mock_client.put_object.call_args_list
    
    success_call = calls[0]
    assert success_call.kwargs["Key"].endswith("_SUCCESS.json")
    success_body = json.loads(success_call.kwargs["Body"])
    assert success_body["file_count"] == 2
    assert success_body["bundle_id"] == pub.bundle_id
    
    latest_call = calls[1]
    assert latest_call.kwargs["Key"] == pub.latest_key
    latest_body = json.loads(latest_call.kwargs["Body"])
    assert latest_body["frames_key"].endswith("frames.parquet")


def test_publish_run_artifacts_smoke_prefix(tmp_path: Path, monkeypatch) -> None:
    _, paths, config = _setup_fixture(tmp_path)
    
    mock_client = MagicMock()
    # Mock out _verify_remote_size for these tests to avoid size mismatch errors
    from hcmai.data.corpus_build import publish
    monkeypatch.setattr(publish, "_verify_remote_size", lambda *args, **kwargs: None)

    pub = publish_run_artifacts(mock_client, paths, config, limit=10)
    
    assert pub.latest_key == "smoke-test-artifacts/latest.json"
    assert pub.version_prefix.startswith(f"smoke-test-artifacts/versions/")


def test_publish_run_artifacts_upload_failure(tmp_path: Path, monkeypatch) -> None:
    _, paths, config = _setup_fixture(tmp_path)
    
    mock_client = MagicMock()
    
    # Real _verify_remote_size will raise because mock_client returns nothing useful
    mock_client.head_object.return_value = {"ContentLength": 0}

    with pytest.raises(OSError, match="Uploaded size mismatch"):
        publish_run_artifacts(mock_client, paths, config, limit=None)
        
    # Since it failed on the first artifact, it should not have called put_object for latest.json
    mock_client.put_object.assert_not_called()
