"""Offline frame-text indexes and online caption/OCR/ASR retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hcmai.common.schemas import RetrievalSource
from hcmai.data import CaptionStore, FrameStore
from hcmai.retriever.dense import DenseIndex, DenseRetriever, TextEncoder

_TEXT_SOURCES = {
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
}


class TextEvidenceRetriever(DenseRetriever):
    """Search a frame-aligned text channel without changing frame identity."""

    def __init__(
        self,
        encoder: TextEncoder,
        index: DenseIndex,
        source: RetrievalSource,
    ) -> None:
        if source not in _TEXT_SOURCES:
            raise ValueError(f"{source.value!r} is not a text evidence source")
        super().__init__(encoder, index, source=source)


class CaptionRetriever(TextEvidenceRetriever):
    def __init__(self, encoder: TextEncoder, index: DenseIndex) -> None:
        super().__init__(encoder, index, RetrievalSource.CAPTION)


class OCRRetriever(TextEvidenceRetriever):
    def __init__(self, encoder: TextEncoder, index: DenseIndex) -> None:
        super().__init__(encoder, index, RetrievalSource.OCR)


class ASRRetriever(TextEvidenceRetriever):
    def __init__(self, encoder: TextEncoder, index: DenseIndex) -> None:
        super().__init__(encoder, index, RetrievalSource.ASR)


def _text_corpus(
    evidence: Any,
    frames: FrameStore,
) -> tuple[list[str], pd.DataFrame]:
    """Join usable enrichment text to canonical frame identity."""

    texts: list[str] = []
    mapping: list[dict[str, Any]] = []
    for row in evidence.iter_records():
        text = evidence.get_text(row.frame_id)
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
        raise ValueError(
            f"{evidence.source.value} artifact contains no usable completed text"
        )
    return texts, pd.DataFrame(mapping)


def _normalized(vectors: np.ndarray) -> np.ndarray:
    """Return finite float32 unit vectors suitable for inner product."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Text encoder must return a finite 2D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Text encoder returned a zero vector")
    return values / norms


def _encode_texts(
    texts: list[str],
    encoder: TextEncoder,
    source: RetrievalSource,
) -> np.ndarray:
    """Encode one text channel with backend-neutral progress reporting."""
    configured = int(getattr(encoder.config, "batch_size", 64))
    batch_size = max(1, min(configured, 64))
    batches: list[np.ndarray] = []
    with tqdm(
        total=len(texts),
        desc=f"Embedding {source.value}",
        unit="text",
        dynamic_ncols=True,
    ) as progress:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batches.append(encoder.encode_text(batch))
            progress.update(len(batch))
    return np.vstack(batches)


def build_text_index(
    evidence: Any,
    frames: FrameStore,
    encoder: TextEncoder,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Encode frame-aligned caption/OCR/ASR text and persist an exact index."""

    source = evidence.source
    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text evidence source {source.value!r}")
    artifact_name = Path(embeddings_filename)
    if (
        artifact_name.name != embeddings_filename
        or artifact_name.suffix != ".npy"
    ):
        raise ValueError("embeddings_filename must be a plain .npy filename")
    texts, mapping = _text_corpus(evidence, frames)
    vectors = _normalized(_encode_texts(texts, encoder, source))
    if len(vectors) != len(mapping):
        raise ValueError(
            f"Text encoder returned {len(vectors)} vectors "
            f"for {len(mapping)} texts"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / artifact_name, vectors)
    index = DenseIndex.build(
        vectors,
        mapping,
        dataset_version=dataset_version,
        model_name=encoder.config.model_name,
        index_type=index_type,
        show_progress=True,
    )
    index.save(output)
    return index


def build_caption_index(
    captions: CaptionStore,
    frames: FrameStore,
    encoder: TextEncoder,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Backward-compatible caption-specific index builder."""

    return build_text_index(
        captions,
        frames,
        encoder,
        output_dir,
        embeddings_filename=embeddings_filename,
        dataset_version=dataset_version,
        index_type=index_type,
    )
