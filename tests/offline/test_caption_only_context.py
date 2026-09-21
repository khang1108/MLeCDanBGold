"""Test deterministic FrameContext build when OCR and Objects are omitted (caption-only)."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from offline.corpus.models import FrameArtifact
from offline.enrichment.caption.models.evidence import CaptionEvidence
from offline.enrichment.context.builder import build_frame_context
from offline.enrichment.context.config import FrameContextConfig
from offline.enrichment.context.models import FrameContext


def test_build_frame_context_caption_only(tmp_path: Path) -> None:
    frames_dir = tmp_path / "artifacts"
    frames_dir.mkdir(parents=True)
    frames_path = frames_dir / "frames.parquet"

    # 1. Write frames.parquet & manifest
    frame_row = {
        "frame_id": "v1_f001",
        "video_id": "v1",
        "frame_idx": 100,
        "keyframe_order": 1,
        "timestamp_ms": 1000,
        "fps": 25.0,
        "image_path": "data/frames/v1/001.jpg",
        "thumbnail_path": None,
        "width": 640,
        "height": 480,
        "shot_id": None,
        "event_id": None,
        "is_anchor": True,
        "pts": None,
        "time_base": None,
        "motion_score": 0.0,
        "shot_score": 0.0,
        "event_score": 0.0,
        "selection_reasons": ["anchor"],
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([frame_row])), frames_path)
    (frames_dir / "manifest.json").write_text(
        json.dumps({"frame_store_id": "v3c-test"}),
        encoding="utf-8",
    )

    # 2. Write captions.parquet & manifest
    caption_dir = tmp_path / "artifacts" / "captions"
    caption_dir.mkdir(parents=True)
    caption_path = caption_dir / "captions.parquet"
    caption_row = {
        "status": "completed",
        "error_code": None,
        "error_message": None,
        "frame_id": "v1_f001",
        "video_id": "v1",
        "frame_idx": 100,
        "timestamp_ms": 1000,
        "text": "A man riding a bicycle on a street",
        "frame_store_id": "v3c-test",
        "artifact_version": "caption-v1",
        "model_name": "Qwen/Qwen3-VL-2B-Instruct",
        "model_revision": None,
    }
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([caption_row])), caption_path)
    (caption_dir / "manifest.json").write_text(
        json.dumps({
            "artifact_version": "caption-v1",
            "frame_store_id": "v3c-test",
        }),
        encoding="utf-8",
    )

    # 3. Build context without OCR or objects
    output_dir = tmp_path / "artifacts" / "context"
    config = FrameContextConfig(
        context_version="frame-context-v1",
        caption_token_budget=80,
        ocr_token_budget=80,
        object_token_budget=40,
        min_ocr_quality=0.5,
    )

    result_path = build_frame_context(
        frames_path=frames_path,
        caption_path=caption_path,
        ocr_frames_path=None,
        object_frames_path=None,
        output_dir=output_dir,
        config=config,
        frame_store_id="v3c-test",
    )

    assert result_path.is_file()
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["context_version"] == "frame-context-v1"
    assert manifest["caption_version"] == "caption-v1"
    assert manifest["ocr_version"] == "none"
    assert manifest["object_version"] == "none"
    assert manifest["frame_store_id"] == "v3c-test"

    table = pq.read_table(result_path)
    records = table.to_pylist()
    assert len(records) == 1
    row = records[0]
    assert row["frame_id"] == "v1_f001"
    assert row["caption_text"] == "A man riding a bicycle on a street"
    assert row["ocr_text"] is None
    assert row["object_summary"] is None
    assert row["caption_available"] is True
    assert row["context_text"] == "[CAPTION]\nA man riding a bicycle on a street"

    # Verify idempotence: rebuilding returns existing path
    rebuilt_path = build_frame_context(
        frames_path=frames_path,
        caption_path=caption_path,
        ocr_frames_path=None,
        object_frames_path=None,
        output_dir=output_dir,
        config=config,
        frame_store_id="v3c-test",
    )
    assert rebuilt_path == result_path
