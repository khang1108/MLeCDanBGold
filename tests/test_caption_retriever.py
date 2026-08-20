"""Smoke test for caption index construction and retrieval."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import EncoderConfig
from hcmai.common.schemas import RetrievalSource
from hcmai.data.pipeline import DataService
from hcmai.data.stores.evidence import ASRStore, CaptionStore, OCRStore
from hcmai.retrieval.retriever.text.retriever import (
    ASRRetriever,
    CaptionRetriever,
    OCRRetriever,
    build_text_index,
)

_CASES = [
    (RetrievalSource.CAPTION, "text", CaptionStore, CaptionRetriever),
    (RetrievalSource.OCR, "normalized_text", OCRStore, OCRRetriever),
    (RetrievalSource.ASR, "asr_text", ASRStore, ASRRetriever),
]


class FakeEncoder:
    """Map fixture captions and queries into one deterministic space."""

    def __init__(self) -> None:
        self.config = EncoderConfig(model_name="fake/text-encoder")
        self.embedding_dim = 2

    def encode_text(self, texts, stats=None) -> np.ndarray:
        vectors = [
            [1.0, 0.0] if "cook" in text.lower() else [0.0, 2.0]
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


@pytest.mark.parametrize(
    ("source", "field", "store_type", "retriever_type"),
    _CASES,
)
def test_text_index_round_trip_and_source_identity(
    tmp_path: Path, source, field, store_type, retriever_type
) -> None:
    frames_path = tmp_path / "frames.parquet"
    evidence_path = tmp_path / f"{source.value}.parquet"
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
    rows: list[dict[str, object]] = []
    for index, text in ((1, "A cook holds a pan."), (2, "A dog runs outside.")):
        row: dict[str, object] = {
            "frame_id": f"frame-{index}",
            field: text,
            "model_name": "fixture",
        }
        if source != RetrievalSource.ASR:
            row.update(
                {
                    "video_id": "video-1",
                    "frame_idx": index,
                    "timestamp_ms": index * 1000,
                    "artifact_version": f"{source.value}-v1",
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(evidence_path, index=False)

    data = DataService.load(frames_path, {source: evidence_path})
    output = tmp_path / f"{source.value}-index"
    index = build_text_index(
        data, FakeEncoder(), source, output,
        embeddings_filename=f"{source.value}_embeddings.npy",
        dataset_version="fixture-v1",
    )
    result = retriever_type(FakeEncoder(), index).search("cook", top_k=1)[0]

    assert result.frame_id == "frame-1"
    assert result.source_ranks == {source: 1}
    assert result.source_scores[source] == 1.0
    assert (output / f"{source.value}_embeddings.npy").is_file()
    assert (output / "dense.index").is_file()
