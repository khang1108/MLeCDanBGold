"""Build fielded BM25 artifacts over canonical frame documents.

This offline module joins title, caption, OCR, and frame-ASR evidence without
rewriting corpus text. It does not load artifacts for online search.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp
from typing import cast

import numpy as np
import pandas as pd
from hcmai.retrieval.retriever.artifacts import publish_directory
from scipy import sparse

FIELDS = ("title", "caption", "ocr", "asr")
IDENTITY_COLUMNS = ("frame_id", "video_id", "frame_idx", "timestamp_ms")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Tokenize text with Unicode NFKC, lowercase, and punctuation boundaries."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    separated = "".join(character if character.isalnum() else " " for character in normalized)
    return separated.split()


def build_bm25_index(
    *,
    frames_path: str | Path,
    caption_path: str | Path,
    ocr_path: str | Path,
    asr_path: str | Path,
    media_info_path: str | Path,
    output_dir: str | Path,
    dataset_version: str,
) -> Path:
    """Build and atomically publish four BM25 document-term matrices."""

    frames = pd.read_parquet(frames_path)
    missing = set(IDENTITY_COLUMNS) - set(frames.columns)

    if missing:
        raise ValueError(f"canonical frames are missing columns: {sorted(missing)}")

    mapping = cast(
        pd.DataFrame,
        frames.loc[:, list(IDENTITY_COLUMNS)].copy().reset_index(drop=True),
    )
    if mapping["frame_id"].duplicated().any():
        raise ValueError("canonical frames contain duplicate frame_id values")

    documents = pd.DataFrame({"frame_id": mapping["frame_id"]})
    documents["title"] = _titles(mapping, Path(media_info_path))
    documents["caption"] = _evidence_text(caption_path, "text", mapping)
    documents["ocr"] = _evidence_text(ocr_path, "normalized_text", mapping, fallback="raw_text")
    documents["asr"] = _evidence_text(asr_path, "asr_text", mapping)

    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))

    try:
        mapping.to_parquet(staged / "frame_mapping.parquet", index=False)
        field_metadata: dict[str, dict[str, int]] = {}

        for field in FIELDS:
            vocabulary, matrix = _build_field(documents[field].tolist())
            (staged / f"{field}_vocab.json").write_text(
                json.dumps(vocabulary, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            sparse.save_npz(staged / f"{field}_weights.npz", matrix)
            field_metadata[field] = {"vocabulary_size": len(vocabulary)}

        metadata = {
            "schema_version": "frame-bm25-v1",
            "tokenizer_version": "nfkc-alnum-v1",
            "dataset_version": dataset_version,
            "document_count": len(mapping),
            "k1": K1,
            "b": B,
            "fields": field_metadata,
        }

        (staged / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

        return publish_directory(staged, destination)
    except Exception:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
        raise


def _titles(mapping: pd.DataFrame, media_info_path: Path) -> list[str]:
    """Resolve organizer titles by canonical video identity."""

    titles: dict[str, str] = {}
    if media_info_path.is_dir():
        for path in sorted(media_info_path.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            title = value.get("title") if isinstance(value, dict) else None
            titles[path.stem] = title.strip() if isinstance(title, str) else ""
    return [titles.get(str(video_id), "") for video_id in mapping["video_id"]]


def _evidence_text(
    path: str | Path,
    column: str,
    mapping: pd.DataFrame,
    *,
    fallback: str | None = None,
) -> list[str]:
    """Left-join one optional evidence artifact to canonical frame order."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        return [""] * len(mapping)
    table = pd.read_parquet(artifact_path)
    if "frame_id" not in table.columns:
        raise ValueError(f"evidence artifact lacks frame_id: {artifact_path}")
    if table["frame_id"].duplicated().any():
        raise ValueError(f"evidence artifact has duplicate frame_id: {artifact_path}")
    values = table[column] if column in table.columns else pd.Series([None] * len(table))
    if fallback is not None and fallback in table.columns:
        values = values.fillna(table[fallback])
    lookup = dict(zip(table["frame_id"].astype(str), values, strict=True))
    return [_clean_text(lookup.get(str(frame_id))) for frame_id in mapping["frame_id"]]


def _clean_text(value: object) -> str:
    """Convert missing evidence to empty text without rewriting usable values."""

    return value.strip() if isinstance(value, str) else ""


def _build_field(texts: list[str]) -> tuple[dict[str, int], sparse.csr_matrix]:
    """Precompute BM25 document-term contributions for one independent field."""

    token_rows = [tokenize(text) for text in texts]
    vocabulary = {
        token: index
        for index, token in enumerate(sorted({token for row in token_rows for token in row}))
    }

    document_count = len(token_rows)
    document_frequency = Counter(token for row in token_rows for token in set(row))

    lengths = np.asarray([len(row) for row in token_rows], dtype=np.float64)
    average_length = float(lengths.mean()) if document_count else 0.0

    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []

    for document_index, tokens in enumerate(token_rows):

        length_ratio = lengths[document_index] / average_length if average_length else 0.0
        for token, frequency in Counter(tokens).items():
            df = document_frequency[token]
            inverse_document_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))

            denominator = frequency + K1 * (1 - B + B * length_ratio)

            weight = inverse_document_frequency * frequency * (K1 + 1) / denominator

            rows.append(document_index)
            columns.append(vocabulary[token])
            data.append(weight)
    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, columns)),
        shape=(document_count, len(vocabulary)),
        dtype=np.float32,
    )
    return vocabulary, matrix


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the canonical production paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="artifacts/frame_store/frames.parquet")
    parser.add_argument("--caption", default="artifacts/corpus/caption.parquet")
    parser.add_argument("--ocr", default="artifacts/corpus/ocr_frames.parquet")
    parser.add_argument("--asr", default="artifacts/enrichment/asr/frame_enrichment.parquet")
    parser.add_argument("--media-info", default="data/media-info")
    parser.add_argument("--output", default="artifacts/indexes/bm25")
    parser.add_argument("--dataset-version", default="hcmai2026_v1")
    return parser


def main() -> None:
    """Build a BM25 artifact from command-line paths."""

    arguments = _parser().parse_args()
    build_bm25_index(
        frames_path=arguments.frames,
        caption_path=arguments.caption,
        ocr_path=arguments.ocr,
        asr_path=arguments.asr,
        media_info_path=arguments.media_info,
        output_dir=arguments.output,
        dataset_version=arguments.dataset_version,
    )


if __name__ == "__main__":
    main()