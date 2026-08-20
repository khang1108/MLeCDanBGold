from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.data.corpus_build import (
    PreparationPaths,
    S3CorpusPreparationConfig,
)
from hcmai.data.corpus_build.pipeline import DefaultPreparationOperations
from hcmai.retrieval.retriever.dense.index import DenseIndex


class FakeVisualEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.embedding_dim = 3

    def encode_images(self, images, stats=None) -> np.ndarray:
        if stats:
            stats.num_encoded += len(images)
            stats.embedding_dim = self.embedding_dim
        # Return valid L2-normalized vectors (length 1)
        vectors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype="float32"), (len(images), 1))
        return vectors


class FakeTextEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.embedding_dim = 4

    def encode_text(self, texts: list[str], stats=None) -> np.ndarray:
        if stats:
            stats.num_encoded += len(texts)
            stats.embedding_dim = self.embedding_dim
        # Return valid L2-normalized vectors
        vectors = np.tile(np.array([[0.0, 1.0, 0.0, 0.0]], dtype="float32"), (len(texts), 1))
        return vectors


def _config(tmp_path: Path) -> S3CorpusPreparationConfig:
    work = (tmp_path / "run").resolve()
    model_names = {
        "dino": "fixture/dino",
        "caption": "fixture/caption",
        "ocr": "fixture/ocr",
        "asr": "fixture/asr",
        "diarization": "fixture/diarization",
        "visual_embedding": "fixture/visual",
        "text_embedding": "fixture/text",
    }
    return S3CorpusPreparationConfig.model_validate({
        "corpus_revision": "fixture-v1",
        "work_root": work,
        "models": {
            name: {"model_name": model, "revision": "a" * 40}
            for name, model in model_names.items()
        },
        "preprocessing": {
            "s3": {
                "bucket": "hcmai-dataset",
                "videos_prefix": "videos",
                "artifacts_prefix": "artifacts/full",
                "smoke_artifacts_prefix": "artifacts/smoke",
                "staging_root": work / "staging",
            },
            "output_root": work / "artifacts/frame_store",
            "transnet_repo": work / "models/transnet",
            "transnet_weights": work / "models/transnet-weights",
            "efficientgebd_repo": work / "models/gebd",
            "efficientgebd_config": work / "models/gebd.yaml",
            "efficientgebd_checkpoint": work / "models/gebd.pth",
            "dino_model": model_names["dino"],
            "dino_revision": "a" * 40,
        },
    })


def _setup_fixture(tmp_path: Path, paths: PreparationPaths) -> None:
    # 1. Create canonical frames
    paths.frame_store_root.mkdir(parents=True, exist_ok=True)
    images = paths.frame_store_root / "keyframes" / "L21_V001"
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
            "width": 8,
            "height": 6,
        }
        for order in (1, 2)
    ]
    pd.DataFrame(rows).to_parquet(paths.frames_path, index=False)
    
    # 2. Create authoritative Caption/OCR evidence and the ASR compatibility view.
    for source in (RetrievalSource.CAPTION, RetrievalSource.OCR, RetrievalSource.ASR):
        enrichment_root = paths.enrichment_path(source).parent
        enrichment_root.mkdir(parents=True, exist_ok=True)
        if source is RetrievalSource.ASR:
            enrichment = [
                {
                    "frame_id": f"frame-{order}",
                    "asr_text": f"asr text {order}",
                    "enrichment_version": "asr-v1",
                    "model_name": "fixture/asr",
                }
                for order in (1, 2)
            ]
        else:
            text_field = (
                "text"
                if source is RetrievalSource.CAPTION
                else "normalized_text"
            )
            enrichment = [
                {
                    "frame_id": f"frame-{order}",
                    "video_id": "L21_V001",
                    "frame_idx": order * 90,
                    "timestamp_ms": order * 3000,
                    text_field: f"{source.value} text {order}",
                    "artifact_version": f"{source.value}-v1",
                    "model_name": f"fixture/{source.value}",
                }
                for order in (1, 2)
            ]
        pd.DataFrame(enrichment).to_parquet(paths.enrichment_path(source), index=False)


def test_four_aligned_indexes_exactly_match_canonical_frame_identities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    paths = PreparationPaths.from_config(config, None)
    _setup_fixture(tmp_path, paths)
    
    # Mock AppConfig and LLMServiceConfig to avoid FileNotFoundError reading non-existent YAMLs
    from hcmai.common.config import AppConfig, TranscriptJobConfig
    from hcmai.data.enrichment.caption.config import CaptionJobConfig
    from hcmai.llm.pipeline import LLMServiceConfig
    
    # Wait, instead of guessing AppConfig shape, let's just mock `AppConfig.from_yaml` to return a mocked object
    # using unittest.mock.MagicMock.
    from unittest.mock import MagicMock
    mock_app_config = MagicMock()
    mock_app_config.dataset.version = "fixture-v1"
    mock_app_config.index.type = "flat_ip"
    # Mock text_embedding_filenames
    mock_app_config.index.text_embedding_filenames = {
        RetrievalSource.CAPTION: "caption_embeddings.npy",
        RetrievalSource.OCR: "ocr_embeddings.npy",
        RetrievalSource.ASR: "asr_embeddings.npy",
    }
    monkeypatch.setattr(AppConfig, "from_yaml", lambda *args, **kwargs: mock_app_config)
    
    mock_models_config = MagicMock()
    mock_models_config.visual_embedding = EncoderConfig(
        model_name="fixture/visual",
        revision="a" * 40,
    )
    mock_models_config.caption_embedding = EncoderConfig(
        model_name="fixture/text",
        revision="a" * 40,
    )
    monkeypatch.setattr(LLMServiceConfig, "from_yaml", lambda *args, **kwargs: mock_models_config)

    mock_caption_job = MagicMock()
    mock_caption_job.caption.model_checkpoint = "fixture/caption"
    mock_caption_job.caption.revision = "a" * 40
    monkeypatch.setattr(
        CaptionJobConfig,
        "from_yaml",
        lambda *args, **kwargs: mock_caption_job,
    )

    mock_transcript_job = MagicMock()
    mock_transcript_job.asr.model_name = "fixture/asr"
    mock_transcript_job.asr.revision = "a" * 40
    mock_transcript_job.diarization.model_name = "fixture/diarization"
    mock_transcript_job.diarization.revision = "a" * 40
    monkeypatch.setattr(
        TranscriptJobConfig,
        "from_yaml",
        lambda *args, **kwargs: mock_transcript_job,
    )

    ops = DefaultPreparationOperations(
        config,
        paths,
        resume=True,
        limit=None,
    )
    
    # Mock the text encoder
    text_config = EncoderConfig(model_name="fixture/text")
    mock_text_encoder = FakeTextEncoder(text_config)
    ops._text_encoder = mock_text_encoder
    
    # Mock the visual encoder 
    # DefaultPreparationOperations uses EmbeddingService.build_visual_artifacts which instantiates SigLIPAdapter by default if encoder is None
    # Let's mock EmbeddingService.build_visual_artifacts to use our FakeVisualEncoder
    from hcmai.retrieval.embedding.pipeline import EmbeddingService
    original_build_visual = EmbeddingService.build_visual_artifacts
    
    def mock_build_visual(*args, **kwargs):
        kwargs["encoder"] = FakeVisualEncoder(EncoderConfig(model_name="fixture/visual"))
        return original_build_visual(*args, **kwargs)
        
    monkeypatch.setattr(EmbeddingService, "build_visual_artifacts", staticmethod(mock_build_visual))

    # Build all 4 indexes
    visual_index_path = ops.build_visual_index()
    caption_index_path = ops.build_text_index(RetrievalSource.CAPTION)
    ocr_index_path = ops.build_text_index(RetrievalSource.OCR)
    asr_index_path = ops.build_text_index(RetrievalSource.ASR)

    # 1. Load canonical frame identities
    canonical_frames = pd.read_parquet(paths.frames_path)
    expected_frame_ids = canonical_frames["frame_id"].tolist()
    expected_video_ids = canonical_frames["video_id"].tolist()

    # 2. Verify all 4 indexes
    for index_path in (
        visual_index_path,
        caption_index_path,
        ocr_index_path,
        asr_index_path,
    ):
        index_dir = index_path.parent
        # load() automatically performs strict validation of row counts, dimensions, and mapping integrity
        index = DenseIndex.load(index_dir)
        
        # Verify alignment: the mappings must EXACTLY match the canonical frame identities in order
        actual_frame_ids = index.mapping["frame_id"].tolist()
        actual_video_ids = index.mapping["video_id"].tolist()
        
        assert actual_frame_ids == expected_frame_ids, f"Frame ID mismatch in {index_dir.name}"
        assert actual_video_ids == expected_video_ids, f"Video ID mismatch in {index_dir.name}"
        
        # Verify metadata
        assert index.metadata.vector_count == len(expected_frame_ids)
        assert index.index.ntotal == len(expected_frame_ids)

        if index_path == visual_index_path:
            assert index.metadata.embedding_dim == 3
        else:
            assert index.metadata.embedding_dim == 4
