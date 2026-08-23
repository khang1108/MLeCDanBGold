"""Smoke test for the executable caption-index pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.common.utils.io import write_yaml
from hcmai.retrieval.retriever.pipeline import RetrievalService


class FakeCaptionEncoder:
    config = EncoderConfig(model_name="fake/caption-encoder")
    embedding_dim = 2

    def encode_text(self, texts, stats=None) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0] if "cook" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_build_text_artifacts_from_two_configs(tmp_path: Path) -> None:
    frames = tmp_path / "frames.parquet"
    captions = tmp_path / "captions.parquet"
    output = tmp_path / "caption-index"
    pipeline_config = tmp_path / "baseline.yaml"
    model_config = tmp_path / "llm.yaml"
    pd.DataFrame(
        [
            {
                "frame_id": "frame-1",
                "video_id": "video-1",
                "frame_idx": 10,
                "timestamp_ms": 1000,
                "image_path": "frame-1.jpg",
                "width": 10,
                "height": 10,
            }
        ]
    ).to_parquet(frames, index=False)
    pd.DataFrame(
        [
            {
                "frame_id": "frame-1",
                "video_id": "video-1",
                "frame_idx": 10,
                "timestamp_ms": 1000,
                "text": "A cook holds a pan.",
                "artifact_version": "caption-v1",
                "model_name": "fake/caption-generator",
            }
        ]
    ).to_parquet(captions, index=False)
    write_yaml(
        {
            "dataset": {
                "version": "fixture-v1",
                "frames_path": str(frames),
                "enrichment": {"caption_path": str(captions)},
            },
            "index": {"caption_path": str(output)},
        },
        pipeline_config,
    )
    write_yaml(
        {
            "visual_embedding": {"name": "fake/visual-encoder"},
            "caption_embedding": {"name": "fake/caption-encoder"},
        },
        model_config,
    )

    index = RetrievalService.build_text_artifacts(
        pipeline_config,
        model_config,
        source=RetrievalSource.CAPTION,
        encoder=FakeCaptionEncoder(),
    )

    assert index.metadata.dataset_version == "fixture-v1"
    assert index.metadata.model_name == "fake/caption-encoder"
    assert index.metadata.vector_count == 1
    assert (output / "caption_embeddings.npy").is_file()
    assert (output / "dense.index").is_file()
    assert (output / "frame_mapping.parquet").is_file()
    assert (output / "metadata.json").is_file()
