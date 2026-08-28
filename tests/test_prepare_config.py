"""Validate the single checked-in configuration for offline preparation.

This module verifies section routing and cross-stage identity. It does not run
model inference, access S3, or build corpus artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import pytest

from hcmai.common.config import TranscriptJobConfig
from hcmai.common.utils.io import read_yaml
from hcmai.data.enrichment.caption.config import CaptionJobConfig
from hcmai.data.enrichment.dataset_cli import add_dataset_arguments, dataset_overrides
from hcmai.data.enrichment.pipeline import EnrichmentJobConfig
from hcmai.data.s3 import load_s3_config
from scripts.build_retrieval_indexes import (
    _index_dataset_overrides,
    load_model_config,
    load_offline_config,
    parse_args as parse_index_args,
)


DATASET = {
    "version": "dataset_v1",
    "source": "custom_raw_video",
    "data_root": "runs/dataset_v1",
    "frame_store_id": "dataset_1",
    "frames_path": "artifacts/dataset_v1/frame_store/frames.parquet",
    "frame_store_output": "artifacts/dataset_v1/frame_store",
}

INDEX_DATASET = {
    **DATASET,
    "frame_manifest": "artifacts/dataset_v1/frame_store/manifest.json",
    "keyframes_root": "data/keyframes",
    "map_keyframes_root": "data/map_keyframes",
    "context_path": "artifacts/dataset_v1/enrichment/context/frame_context_v1.parquet",
    "transcripts_path": "artifacts/dataset_v1/enrichment/transcripts",
    "expected_video_count": 873,
    "expected_frame_count": 177321,
}

CUSTOM_INDEX_DATASET = {
    **DATASET,
    "source": "custom_raw_video_1fps",
    "frame_manifest": "artifacts/dataset_v1/frame_store/manifest.json",
    "context_path": "artifacts/dataset_v1/enrichment/context/frame_context_v1.parquet",
    "transcripts_path": "artifacts/dataset_v1/enrichment/transcripts",
    "expected_video_count": 10,
    "expected_frame_count": 1_000,
}


def test_prepare_yaml_routes_every_offline_stage_from_one_file() -> None:
    """Load enrichment, storage, indexing, and model contracts together."""

    path = Path("configs/prepare.yaml")
    raw = read_yaml(path)
    assert set(raw) == {"enrichment", "storage", "indexing", "models", "custom_pipeline"}
    assert "dataset" not in raw["enrichment"]
    assert "dataset" not in raw["indexing"]

    enrichment = EnrichmentJobConfig.from_yaml(path, dataset=DATASET)
    caption = CaptionJobConfig.from_yaml(path, dataset=DATASET)
    transcript = TranscriptJobConfig.from_yaml(path, dataset=DATASET)
    storage = load_s3_config(path)
    indexing = load_offline_config(path, path, dataset=INDEX_DATASET)
    models = load_model_config(path)

    assert enrichment.dataset_version == INDEX_DATASET["version"]
    assert enrichment.frame_store_id == INDEX_DATASET["frame_store_id"]
    assert caption.caption.dataset_version == indexing.dataset.version
    assert transcript.frames_path == enrichment.frames_path
    assert transcript.frame_store_id == enrichment.frame_store_id
    assert indexing.dataset.context_path.name == "frame_context_v1.parquet"
    assert storage.bucket == "mlecdanbgold-hcmai-hk"
    assert enrichment.objects.batch_size == 32
    assert models.visual_embedding.batch_size == 128
    assert models.resolved_evidence_embedding.batch_size == 128


def test_custom_index_config_does_not_require_btc_mapping() -> None:
    """A native custom FrameStore is already authoritative for coordinates."""

    indexing = load_offline_config(
        "configs/prepare.yaml",
        "configs/prepare.yaml",
        dataset=CUSTOM_INDEX_DATASET,
    )

    assert indexing.dataset.uses_btc_mapping is False
    assert indexing.dataset.keyframes_root is None
    assert indexing.dataset.map_keyframes_root is None
    assert indexing.dataset.visual_root == Path(DATASET["data_root"])

    cli = parse_index_args(
        [
            "--version",
            "custom-v1",
            "--source",
            "custom_raw_video_1fps",
            "--frame-store-id",
            "custom-store-v1",
            "--data-root",
            "runs/custom-v1",
            "--frames",
            "runs/custom-v1/frame_store/frames.parquet",
            "--frame-store-output",
            "runs/custom-v1/frame_store",
            "--frame-manifest",
            "runs/custom-v1/frame_store/manifest.json",
            "--context",
            "runs/custom-v1/enrichment/context/frame_context_v1.parquet",
            "--transcripts",
            "runs/custom-v1/enrichment/transcripts",
            "--expected-video-count",
            "10",
            "--expected-frame-count",
            "1000",
        ]
    )
    values = _index_dataset_overrides(cli)
    assert values is not None
    assert values["keyframes_root"] is None
    assert values["map_keyframes_root"] is None


def test_full_pipeline_extra_has_one_transformers_contract() -> None:
    """The documented single environment must have a satisfiable model stack."""

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    optional = project["project"]["optional-dependencies"]

    assert "transformers>=5.13, <6.0" in optional["embedding"]
    assert "transformers>=5.13, <6.0" in optional["reranking"]
    assert "transformers>=5.13, <6.0" in optional["transcripts"]
    assert "transformers>=5.13, <6.0" in optional["pipeline"]
    assert "yt-dlp[default]>=2026.8.19, <2027.0" in optional["pipeline"]


def test_dataset_contract_is_complete_and_cli_owned() -> None:
    """Prevent a stage from mixing a partial CLI identity with YAML values."""

    parser = argparse.ArgumentParser()
    add_dataset_arguments(parser)
    args = parser.parse_args(
        [
            "--version",
            "fixture-v1",
            "--source",
            "custom_raw_video",
            "--frame-store-id",
            "fixture-store-v1",
            "--data-root",
            "runs/fixture-v1",
            "--frames",
            "runs/fixture-v1/frame_store/frames.parquet",
            "--frame-store-output",
            "runs/fixture-v1/frame_store",
        ]
    )

    values = dataset_overrides(args)

    assert values == {
        "version": "fixture-v1",
        "source": "custom_raw_video",
        "frame_store_id": "fixture-store-v1",
        "data_root": Path("runs/fixture-v1"),
        "frames_path": Path("runs/fixture-v1/frame_store/frames.parquet"),
        "frame_store_output": Path("runs/fixture-v1/frame_store"),
    }

    with pytest.raises(ValueError, match="frame-store-output"):
        dataset_overrides(
            argparse.Namespace(
                dataset_version="fixture-v1",
                source="custom_raw_video",
                frame_store_id="fixture-store-v1",
                data_root=Path("runs/fixture-v1"),
                frames=Path("runs/fixture-v1/frame_store/frames.parquet"),
                frame_store_output=None,
            )
        )

    job = EnrichmentJobConfig.from_yaml(
        "configs/prepare.yaml",
        dataset=values,
    )
    assert job.dataset_version == "fixture-v1"
    assert job.frame_store_id == "fixture-store-v1"
