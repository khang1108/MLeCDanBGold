"""Tests for canonical frame BM25 artifact construction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from offline.indexes.bm25 import build_bm25_index, tokenize
from scipy import sparse


def test_tokenize_keeps_unicode_alphanumeric_tokens() -> None:
    """Normalize Unicode and punctuation without stemming or segmentation."""

    assert tokenize("HTV: 60 Giây Sáng, X!") == ["htv", "60", "giây", "sáng", "x"]


def test_builder_preserves_canonical_mapping_and_missing_fields(tmp_path: Path) -> None:
    """Build four field matrices over every canonical frame."""

    frames = pd.DataFrame(
        [
            {"frame_id": "f1", "video_id": "v1", "frame_idx": 10, "timestamp_ms": 1000},
            {"frame_id": "f2", "video_id": "v1", "frame_idx": 20, "timestamp_ms": 2000},
            {"frame_id": "f3", "video_id": "v2", "frame_idx": 30, "timestamp_ms": 3000},
        ]
    )
    caption = pd.DataFrame([{"frame_id": "f1", "text": "white apron"}])
    ocr = pd.DataFrame([{"frame_id": "f2", "normalized_text": "HTV 60"}])
    asr = pd.DataFrame([{"frame_id": "f3", "asr_text": "xin chao"}])
    frames_path = tmp_path / "frames.parquet"
    caption_path = tmp_path / "caption.parquet"
    ocr_path = tmp_path / "ocr.parquet"
    asr_path = tmp_path / "asr.parquet"
    media_info = tmp_path / "media-info"
    output = tmp_path / "bm25"
    media_info.mkdir()
    frames.to_parquet(frames_path)
    caption.to_parquet(caption_path)
    ocr.to_parquet(ocr_path)
    asr.to_parquet(asr_path)
    (media_info / "v1.json").write_text(json.dumps({"title": "Morning News"}), encoding="utf-8")

    build_bm25_index(
        frames_path=frames_path,
        caption_path=caption_path,
        ocr_path=ocr_path,
        asr_path=asr_path,
        media_info_path=media_info,
        output_dir=output,
        dataset_version="fixture-v1",
    )

    mapping = pd.read_parquet(output / "frame_mapping.parquet")
    assert mapping.to_dict(orient="records") == frames.to_dict(orient="records")
    for field in ("title", "caption", "ocr", "asr"):
        matrix = sparse.load_npz(output / f"{field}_weights.npz")
        assert matrix.format == "csr"
        assert matrix.shape[0] == 3
        assert json.loads((output / f"{field}_vocab.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["document_count"] == 3
    assert metadata["dataset_version"] == "fixture-v1"