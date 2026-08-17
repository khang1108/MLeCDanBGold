"""Offline frame-text indexes and online caption/OCR/ASR retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hcmai.common.schemas import RetrievalSource
from hcmai.data.pipeline import DataService
from hcmai.retrieval.embedding.pipeline import TextEmbeddingAdapter
from hcmai.retrieval.retriever.dense.index import DenseIndex
from hcmai.retrieval.retriever.dense.retriever import DenseRetriever
from hcmai.retrieval.retriever.cache import EmbeddingCache

_TEXT_SOURCES = {
    RetrievalSource.CAPTION,
    RetrievalSource.OCR,
    RetrievalSource.ASR,
}


class TextEvidenceRetriever(DenseRetriever):
    """Lớp cơ sở (Base class) cho các Retriever văn bản (như Caption, OCR, ASR).
    Sử dụng FAISS (qua DenseRetriever) để tìm kiếm các đoạn text liên quan nhất đến câu truy vấn,
    sau đó trả về kết quả kèm theo thông tin frame (video_id, frame_idx).
    """

    def __init__(
        self,
        encoder: TextEmbeddingAdapter,
        index: DenseIndex,
        source: RetrievalSource,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        if source not in _TEXT_SOURCES:
            raise ValueError(f"{source.value!r} is not a text evidence source")
        super().__init__(
            encoder,
            index,
            source=source,
            embedding_cache=embedding_cache,
            prompt_version=prompt_version,
        )


class CaptionRetriever(TextEvidenceRetriever):
    def __init__(
        self, encoder: TextEmbeddingAdapter, index: DenseIndex,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        super().__init__(encoder, index, RetrievalSource.CAPTION, embedding_cache, prompt_version)


class OCRRetriever(TextEvidenceRetriever):
    def __init__(
        self, encoder: TextEmbeddingAdapter, index: DenseIndex,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        super().__init__(encoder, index, RetrievalSource.OCR, embedding_cache, prompt_version)


class ASRRetriever(TextEvidenceRetriever):
    def __init__(
        self, encoder: TextEmbeddingAdapter, index: DenseIndex,
        embedding_cache: EmbeddingCache | None = None,
        prompt_version: str = "query-v1",
    ) -> None:
        super().__init__(encoder, index, RetrievalSource.ASR, embedding_cache, prompt_version)


def _text_corpus(
    data: DataService,
    source: RetrievalSource,
) -> tuple[list[str], pd.DataFrame]:
    """Join usable enrichment text to canonical frame identity."""

    texts: list[str] = []
    mapping: list[dict[str, Any]] = []
    for row in data.iter_evidence(source):
        frame_id = str(getattr(row, "frame_id"))
        text = data.get_evidence(frame_id, source)
        if text is None:
            continue
        frame = data.get_frame(frame_id)
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
            f"{source.value} artifact contains no usable completed text"
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
    encoder: TextEmbeddingAdapter,
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
    data: DataService,
    encoder: TextEmbeddingAdapter,
    source: RetrievalSource,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Đọc dữ liệu (text) từ DataService, gọi encoder để trích xuất vector đặc trưng,
    và xây dựng (build) một index hoàn chỉnh (bao gồm metadata và vector index) rồi lưu xuống đĩa.
    """

    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text evidence source {source.value!r}")
    artifact_name = Path(embeddings_filename)
    if (
        artifact_name.name != embeddings_filename
        or artifact_name.suffix != ".npy"
    ):
        raise ValueError("embeddings_filename must be a plain .npy filename")
    texts, mapping = _text_corpus(data, source)
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


def build_text_embedding_artifacts(
    data: DataService,
    encoder: TextEmbeddingAdapter,
    source: RetrievalSource,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
) -> tuple[Path, Path]:
    """Persist deterministic frame-aligned text vectors without an index."""

    if source not in _TEXT_SOURCES:
        raise ValueError(f"Unsupported text evidence source {source.value!r}")
    artifact_name = Path(embeddings_filename)
    if artifact_name.name != embeddings_filename or artifact_name.suffix != ".npy":
        raise ValueError("embeddings_filename must be a plain .npy filename")
    texts, mapping = _text_corpus(data, source)
    vectors = _normalized(_encode_texts(texts, encoder, source))
    if len(vectors) != len(mapping):
        raise ValueError("text embedding count does not match mapping rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors_path = output / artifact_name
    mapping_path = output / "frame_mapping.parquet"
    vectors_partial = vectors_path.with_suffix(f"{vectors_path.suffix}.partial")
    mapping_partial = mapping_path.with_suffix(f"{mapping_path.suffix}.partial")
    try:
        with vectors_partial.open("wb") as handle:
            np.save(handle, vectors)
        mapping.to_parquet(mapping_partial, index=False)
        vectors_partial.replace(vectors_path)
        mapping_partial.replace(mapping_path)
    finally:
        vectors_partial.unlink(missing_ok=True)
        mapping_partial.unlink(missing_ok=True)
    return vectors_path, mapping_path


def build_caption_index(
    data: DataService,
    encoder: TextEmbeddingAdapter,
    output_dir: str | Path,
    *,
    embeddings_filename: str,
    dataset_version: str,
    index_type: str = "flat_ip",
) -> DenseIndex:
    """Backward-compatible caption-specific index builder."""

    return build_text_index(
        data,
        encoder,
        RetrievalSource.CAPTION,
        output_dir,
        embeddings_filename=embeddings_filename,
        dataset_version=dataset_version,
        index_type=index_type,
    )
