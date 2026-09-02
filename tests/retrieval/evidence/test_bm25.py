"""Tests for field-routed runtime BM25 temporal scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hcmai.common.config import BM25FieldWeights
from hcmai.retrieval.evidence.bm25 import BM25ArtifactError, BM25TemporalScorer
from offline.indexes.bm25 import build_bm25_index


def _artifact(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    canonical = pd.DataFrame([
        {"frame_id": "f1", "video_id": "v1", "frame_idx": 1, "timestamp_ms": 100},
        {"frame_id": "f2", "video_id": "v1", "frame_idx": 2, "timestamp_ms": 200},
        {"frame_id": "f3", "video_id": "v2", "frame_idx": 3, "timestamp_ms": 300},
    ])
    frames_path = tmp_path / "frames.parquet"
    caption_path = tmp_path / "caption.parquet"
    ocr_path = tmp_path / "ocr.parquet"
    asr_path = tmp_path / "asr.parquet"
    media_info = tmp_path / "media-info"
    output = tmp_path / "bm25"
    media_info.mkdir()
    canonical.to_parquet(frames_path)
    pd.DataFrame([{"frame_id": "f1", "text": "white apron"}]).to_parquet(caption_path)
    pd.DataFrame([{"frame_id": "f2", "normalized_text": "HTV"}]).to_parquet(ocr_path)
    pd.DataFrame([{"frame_id": "f3", "asr_text": "xin chao"}]).to_parquet(asr_path)
    (media_info / "v2.json").write_text(json.dumps({"title": "HTV News"}), encoding="utf-8")
    build_bm25_index(
        frames_path=frames_path,
        caption_path=caption_path,
        ocr_path=ocr_path,
        asr_path=asr_path,
        media_info_path=media_info,
        output_dir=output,
        dataset_version="fixture",
    )
    return output, canonical


def test_language_routing_scores_vi_fields_and_english_caption(tmp_path: Path) -> None:
    """Keep original VI fields separate from selected English caption text."""

    artifact, canonical = _artifact(tmp_path)
    scorer = BM25TemporalScorer.load(artifact, canonical, BM25FieldWeights())

    scores = scorer.score_events(("HTV",), ("white apron",))
    no_caption = scorer.score_events(("white apron",), ("unknown",))

    assert scores.shape == (1, 3)
    assert scores.dtype == np.float32
    assert scores[0, 0] > 0
    assert scores[0, 1] > 0
    assert scores[0, 2] > 0
    np.testing.assert_array_equal(no_caption, np.zeros((1, 3), dtype=np.float32))


def test_identity_mismatch_is_rejected_before_scoring(tmp_path: Path) -> None:
    """Reject stale BM25 artifacts even when frame IDs still match."""

    artifact, canonical = _artifact(tmp_path)
    canonical.loc[1, "frame_idx"] = 999

    with pytest.raises(BM25ArtifactError, match="identity"):
        BM25TemporalScorer.load(artifact, canonical, BM25FieldWeights())


def test_incompatible_tokenizer_metadata_is_rejected(tmp_path: Path) -> None:
    artifact, canonical = _artifact(tmp_path)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tokenizer_version"] = "unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(BM25ArtifactError, match="incompatible"):
        BM25TemporalScorer.load(artifact, canonical, BM25FieldWeights())