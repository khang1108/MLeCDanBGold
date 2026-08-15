"""Tests for reproducible model setup."""

import hashlib
from pathlib import Path

import yaml

from hcmai.data.preprocessing import PreprocessingConfig
from hcmai.data.setup_models import _download, _write_preprocessing_config


def test_download_is_atomic_and_checksum_verified(tmp_path: Path) -> None:
    """Download a valid file and reuse it during offline verification."""

    source = tmp_path / "source.bin"
    source.write_bytes(b"official model weight")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "models/weight.bin"

    _download(source.as_uri(), target, digest, False)
    _download("https://invalid.example", target, digest, True)

    assert target.read_bytes() == source.read_bytes()
    assert not target.with_suffix(".bin.partial").exists()


def test_generated_preprocessing_config_uses_downloaded_models(
    tmp_path: Path,
) -> None:
    """Generate paths accepted by the current preprocessing config."""

    sources = Path("src/hcmai/data/model_sources.yaml")
    settings = yaml.safe_load(sources.read_text(encoding="utf-8"))
    output = tmp_path / "preprocessing.yaml"

    _write_preprocessing_config(tmp_path, settings["preprocessing"], output)
    config = PreprocessingConfig.from_yaml(output)

    assert config.transnet_repo == (tmp_path / "TransNetV2").resolve()
    assert config.efficientgebd_checkpoint == (
        tmp_path / "EfficientGEBD/output/x2x3x4_r50_eff/model_best.pth"
    ).resolve()
    assert config.dino_model == "facebook/dinov2-small"
