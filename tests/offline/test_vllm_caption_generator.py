"""Integration test for caption generator with VLLMCaptionAdapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from offline.enrichment.caption.adapters.vllm import VLLMCaptionAdapter
from offline.enrichment.caption.config import CaptionConfig
from offline.enrichment.caption.generator import generate_captions


def test_REQ_004_generate_captions_with_vllm_adapter(tmp_path: Path) -> None:
    """Generate caption parquet and manifest artifacts using VLLMCaptionAdapter."""
    # Create dummy frame image
    img_path = tmp_path / "frame_001.jpg"
    Image.new("RGB", (32, 32), (0, 255, 0)).save(img_path)

    # Create dummy manifest frames parquet
    frames_table = pa.Table.from_pydict({
        "frame_id": ["v001_f001"],
        "video_id": ["v001"],
        "frame_idx": [1],
        "timestamp_ms": [1000],
        "image_path": [str(img_path)],
    })
    frames_file = tmp_path / "frames.parquet"
    pq.write_table(frames_table, frames_file)

    output_dir = tmp_path / "output"
    config = CaptionConfig(
        model_checkpoint="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="test-rev",
        prompt="qwen vl",
        decoding={},
        device="cuda",
        precision="bfloat16",
        dtype="bfloat16",
        image_size=32,
        batch_size=1,
        enrichment_version="v1",
        write_interval=10,
        dataset_version="v1",
    )
    adapter = VLLMCaptionAdapter(
        config=config,
        batch_fn=lambda imgs: ["A green square frame in clear focus."],
    )

    manifest = generate_captions(
        frames_path=frames_file,
        output_dir=output_dir,
        config=config,
        captioner=adapter,
        dataset_root=tmp_path,
    )

    assert manifest["completed_count"] == 1
    assert manifest["failed_count"] == 0

    captions_pq = output_dir / "captions.parquet"
    assert captions_pq.exists()
    table = pq.read_table(captions_pq)
    assert table.num_rows == 1
    assert table.column("text")[0].as_py() == "A green square frame in clear focus."
    assert table.column("video_id")[0].as_py() == "v001"
    assert table.column("frame_idx")[0].as_py() == 1
