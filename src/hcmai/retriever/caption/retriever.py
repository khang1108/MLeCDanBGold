"""Offline caption-index construction and online caption retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcmai.common.schemas import RetrievalSource
from hcmai.data import CaptionStore, FrameStore
from hcmai.retriever.dense import DenseIndex, DenseRetriever, TextEncoder

CAPTION_EMBEDDINGS_FILENAME = "caption_embeddings.npy"


class CaptionRetriever(DenseRetriever):
    """Search caption embeddings and emit caption-source candidates."""

    def __init__(self, encoder: TextEncoder, index: DenseIndex) -> None:
        super().__init__(encoder, index, source=RetrievalSource.CAPTION)


def _caption_corpus(
    captions: CaptionStore,
    frames: FrameStore,
) -> tuple[list[str], pd.DataFrame]:
    """Join usable captions to canonical frame identity."""

    texts: list[str] = []
    mapping: list[dict[str, Any]] = []
    for row in captions.iter_records():
        text = captions.get_text(row.frame_id)
        if text is None:
            continue
        frame = frames.get(row.frame_id)
        mapping.append(
            {
                "frame_id": frame.frame_id,
                "video_id": frame.video_id,
                "frame_idx": frame.frame_idx,
                "timestamp_ms": frame.timestamp_ms,
                "embedding_index": len(texts),
            }
        )
        texts.append(text)
    if not texts:
        raise ValueError("Caption artifact contains no usable completed captions")
    return texts, pd.DataFrame(mapping)


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Return finite float32 unit vectors suitable for inner product."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Caption encoder must return a finite 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Caption encoder returned a zero vector")
    return values / norms


def build_caption_index(
    captions: CaptionStore,
    frames: FrameStore,
    encoder: TextEncoder,
    output_dir: str | Path,
    *,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Encode captions and persist aligned embeddings plus an exact index."""

    texts, mapping = _caption_corpus(captions, frames)
    vectors = _normalized(encoder.encode_text(texts))
    if len(vectors) != len(mapping):
        raise ValueError(
            f"Caption encoder returned {len(vectors)} vectors "
            f"for {len(mapping)} texts"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / CAPTION_EMBEDDINGS_FILENAME, vectors)
    index = DenseIndex.build(
        vectors,
        mapping,
        dataset_version=dataset_version,
        model_name=encoder.config.model_name,
        index_type=index_type,
    )
    index.save(output)
    return index
