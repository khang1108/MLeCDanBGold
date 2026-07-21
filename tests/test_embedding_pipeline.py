"""Tests for the embedding pipeline."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.embedding.embedding import EmbeddingPipeline
from hcmai.embedding.models.metadata import EmbeddingMetadata


@pytest.fixture
def encoder_config():
    """Create encoder config for testing."""
    return EncoderConfig(
        model_name="google/siglip2-base-patch16-224",
        device="cpu",
        batch_size=2,
    )


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        yield {
            "data": tmpdir / "data",
            "artifacts": tmpdir / "artifacts",
            "logs": tmpdir / "logs",
        }


class TestEmbeddingMetadata:
    """Test EmbeddingMetadata serialization."""

    def test_to_dict(self):
        """Test metadata conversion to dictionary."""
        metadata = EmbeddingMetadata(
            dataset_version="test_v1",
            model_name="siglip2",
            model_checkpoint=None,
            preprocessing_size=224,
            dtype="float32",
            embedding_dimension=768,
            total_frames=100,
            successful_frames=98,
            failed_frames=2,
            normalization="l2",
            generated_at="2026-01-01T00:00:00Z",
            device="cpu",
            batch_size=32,
            processing_time_sec=10.5,
        )
        data = metadata.to_dict()
        assert data["dataset_version"] == "test_v1"
        assert data["total_frames"] == 100
        assert data["successful_frames"] == 98

    def test_from_dict(self):
        """Test metadata creation from dictionary."""
        data = {
            "dataset_version": "test_v1",
            "model_name": "siglip2",
            "model_checkpoint": None,
            "preprocessing_size": 224,
            "dtype": "float32",
            "embedding_dimension": 768,
            "total_frames": 100,
            "successful_frames": 98,
            "failed_frames": 2,
            "normalization": "l2",
            "generated_at": "2026-01-01T00:00:00Z",
            "device": "cpu",
            "batch_size": 32,
            "processing_time_sec": 10.5,
        }
        metadata2 = EmbeddingMetadata.from_dict(data)
        assert metadata2.dataset_version == "test_v1"
        assert metadata2.total_frames == 100


class TestEmbeddingPipeline:
    """Test the EmbeddingPipeline class."""

    def test_pipeline_initialization(self, encoder_config, temp_dirs):
        """Test pipeline initialization."""
        frames_file = temp_dirs["data"] / "frames.parquet"
        frames_file.parent.mkdir(parents=True, exist_ok=True)

        # Create a minimal frames DataFrame
        df = pd.DataFrame({
            "frame_id": ["f001", "f002"],
            "video_id": ["v001", "v001"],
            "frame_idx": [0, 1],
            "timestamp_ms": [0, 33],
            "image_path": [
                str(temp_dirs["data"] / "img001.jpg"),
                str(temp_dirs["data"] / "img002.jpg"),
            ],
            "width": [1920, 1920],
            "height": [1080, 1080],
        })
        df.to_parquet(frames_file)

        pipeline = EmbeddingPipeline(
            frames_path=frames_file,
            output_dir=temp_dirs["artifacts"],
            encoder_config=encoder_config,
            dataset_version="test_v1",
        )

        assert pipeline.frames_path == frames_file
        assert pipeline.output_dir == temp_dirs["artifacts"]
        assert pipeline.dataset_version == "test_v1"

    def test_pipeline_directory_structure(self, encoder_config, temp_dirs):
        """Test that pipeline creates expected directory structure."""
        frames_file = temp_dirs["data"] / "frames.parquet"
        frames_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "frame_id": ["f001"],
            "video_id": ["v001"],
            "frame_idx": [0],
            "timestamp_ms": [0],
            "image_path": [str(temp_dirs["data"] / "img001.jpg")],
            "width": [1920],
            "height": [1080],
        })
        df.to_parquet(frames_file)

        pipeline = EmbeddingPipeline(
            frames_path=frames_file,
            output_dir=temp_dirs["artifacts"],
            encoder_config=encoder_config,
        )

        assert pipeline.embeddings_dir.exists()
        assert pipeline.embeddings_file.parent == pipeline.embeddings_dir
        assert pipeline.mapping_file.parent == pipeline.embeddings_dir

    def test_pipeline_empty_frames(self, encoder_config, temp_dirs):
        """Test pipeline with empty frames list."""
        frames_file = temp_dirs["data"] / "frames.parquet"
        frames_file.parent.mkdir(parents=True, exist_ok=True)

        # Create empty DataFrame
        df = pd.DataFrame({
            "frame_id": [],
            "video_id": [],
            "frame_idx": [],
            "timestamp_ms": [],
            "image_path": [],
            "width": [],
            "height": [],
        })
        df.to_parquet(frames_file)

        pipeline = EmbeddingPipeline(
            frames_path=frames_file,
            output_dir=temp_dirs["artifacts"],
            encoder_config=encoder_config,
        )

        assert pipeline.embeddings_list == []
        assert pipeline.frame_mapping == []

    def test_embedding_save_and_load(self, temp_dirs):
        """Test saving and loading embeddings."""
        embeddings_dir = temp_dirs["artifacts"] / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        embeddings_file = embeddings_dir / "visual_embeddings.npy"

        # Create and save embeddings
        embeddings = np.random.randn(10, 768).astype("float32")
        np.save(embeddings_file, embeddings)

        # Load embeddings
        loaded = np.load(embeddings_file)

        assert loaded.shape == (10, 768)
        np.testing.assert_array_almost_equal(embeddings, loaded)

    def test_checkpoint_save_load(self, encoder_config, temp_dirs):
        """Test checkpoint saving and loading."""
        frames_file = temp_dirs["data"] / "frames.parquet"
        frames_file.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "frame_id": ["f001", "f002"],
            "video_id": ["v001", "v001"],
            "frame_idx": [0, 1],
            "timestamp_ms": [0, 33],
            "image_path": [
                str(temp_dirs["data"] / "img001.jpg"),
                str(temp_dirs["data"] / "img002.jpg"),
            ],
            "width": [1920, 1920],
            "height": [1080, 1080],
        })
        df.to_parquet(frames_file)

        checkpoint_file = temp_dirs["artifacts"] / "checkpoint.json"

        pipeline = EmbeddingPipeline(
            frames_path=frames_file,
            output_dir=temp_dirs["artifacts"],
            encoder_config=encoder_config,
            checkpoint_file=checkpoint_file,
        )

        # Simulate processing some frames
        pipeline.processed_frame_ids = {"f001", "f002"}
        pipeline.save_checkpoint()

        assert checkpoint_file.exists()

        # Load checkpoint
        pipeline2 = EmbeddingPipeline(
            frames_path=frames_file,
            output_dir=temp_dirs["artifacts"],
            encoder_config=encoder_config,
            checkpoint_file=checkpoint_file,
        )
        pipeline2._load_checkpoint()

        assert "f001" in pipeline2.processed_frame_ids
        assert "f002" in pipeline2.processed_frame_ids
