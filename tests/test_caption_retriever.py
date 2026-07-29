"""Smoke test for caption index construction and retrieval."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hcmai.common.schemas import RetrievalSource
from hcmai.data import CaptionStore, FrameStore
from hcmai.retriever.caption import CaptionRetriever, build_caption_index


class FakeEncoder:
    """Map fixture captions and queries into one deterministic space."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(model_name="fake/text-encoder")
        self.embedding_dim = 2

    def encode_text(self, texts, stats=None) -> np.ndarray:
        vectors = [
            [1.0, 0.0] if "cook" in text.lower() else [0.0, 2.0]
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


def test_caption_index_round_trip_and_source_identity(tmp_path: Path) -> None:
    frames_path = tmp_path / "frames.parquet"
    captions_path = tmp_path / "captions.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": f"frame-{index}",
                "video_id": "video-1",
                "frame_idx": index,
                "timestamp_ms": index * 1000,
                "image_path": f"{index}.jpg",
                "width": 10,
                "height": 10,
            }
            for index in (1, 2)
        ]
    ).to_parquet(frames_path, index=False)
    pd.DataFrame(
        [
            {
                "frame_id": "frame-1",
                "caption": "A cook holds a pan.",
                "model_name": "fixture",
            },
            {
                "frame_id": "frame-2",
                "caption": "A dog runs outside.",
                "model_name": "fixture",
            },
        ]
    ).to_parquet(captions_path, index=False)

    index = build_caption_index(
        CaptionStore(captions_path),
        FrameStore(frames_path),
        FakeEncoder(),
        tmp_path / "caption-index",
        dataset_version="fixture-v1",
    )
    result = CaptionRetriever(FakeEncoder(), index).search("cook", top_k=1)[0]

    assert result.frame_id == "frame-1"
    assert result.source_ranks == {RetrievalSource.CAPTION: 1}
    assert result.source_scores[RetrievalSource.CAPTION] == 1.0
    assert (tmp_path / "caption-index/caption_embeddings.npy").is_file()
    assert (tmp_path / "caption-index/dense.index").is_file()
