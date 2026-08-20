from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder
from hcmai.retrieval.embedding.models.metadata import EmbeddingMetadata


class FakeEncoder:
    """Small deterministic encoder that never loads a checkpoint."""

    def __init__(self, _config: EncoderConfig) -> None:
        self.config = _config
        self.embedding_dim = 3

    def encode_images(self, images, stats=None) -> np.ndarray:
        assert stats is not None
        stats.num_encoded += len(images)
        stats.embedding_dim = self.embedding_dim
        return np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype="float32"),
            (len(images), 1),
        )


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    images = root / "keyframes" / "L21_V001"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(images / "001.jpg")
    Image.new("RGB", (8, 6)).save(images / "002.jpg")
    rows = [
        {
            "frame_id": f"frame-{order}",
            "video_id": "L21_V001",
            "frame_idx": order * 90,
            "timestamp_ms": order * 3000,
            "keyframe_order": order,
            "image_path": f"keyframes/L21_V001/{order:03d}.jpg",
        }
        for order in (1, 2, 3)
    ]
    frames_path = tmp_path / "frames.parquet"
    pd.DataFrame(rows).to_parquet(frames_path, index=False)
    return root, frames_path


def test_pipeline_resolves_relative_paths_and_aligns_artifacts(
    tmp_path: Path,
) -> None:
    dataset_root, frames_path = _corpus(tmp_path)
    config = EncoderConfig(batch_size=2)
    pipeline = EmbeddingArtifactBuilder(
        frames_path=frames_path,
        dataset_root=dataset_root,
        output_dir=tmp_path / "artifacts",
        encoder_config=config,
        dataset_version="fixture-v1",
        encoder=FakeEncoder(config),
        strict=False,
    )

    metadata = pipeline.run()
    vectors = np.load(pipeline.embeddings_file)
    mapping = pd.read_parquet(pipeline.mapping_file)

    assert vectors.shape == (2, 3)
    assert mapping["frame_id"].tolist() == ["frame-1", "frame-2"]
    assert mapping["embedding_index"].tolist() == [0, 1]
    assert mapping["keyframe_order"].tolist() == [1, 2]
    assert metadata.total_frames == 3
    assert metadata.successful_frames == 2
    assert metadata.failed_frames == 1
    assert pipeline.metadata_file.is_file()


def test_embedding_metadata_round_trip() -> None:
    values = {
        "dataset_version": "v1", "model_name": "fake",
        "model_checkpoint": None, "preprocessing_size": 224,
        "dtype": "float32", "embedding_dimension": 3,
        "total_frames": 2, "successful_frames": 2, "failed_frames": 0,
        "normalization": "l2", "generated_at": "2026-01-01",
        "device": "cpu", "batch_size": 2, "processing_time_sec": 0.1,
    }
    metadata = EmbeddingMetadata.from_dict(values)
    assert metadata.model_revision is None
    assert metadata.source_fingerprint is None
    assert metadata.schema_version == "visual-embedding-v2"
