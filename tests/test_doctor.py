import json
from pathlib import Path

import pandas as pd
import yaml

from hcmai.common.schemas import FrameRecord
from hcmai.orchestration.diagnostics import diagnose


def test_doctor_validates_assets_and_index_alignment(tmp_path: Path) -> None:
    image = tmp_path / "keyframes" / "v1" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    metadata = tmp_path / "frames.parquet"
    frame = FrameRecord(
        frame_id="f1",
        video_id="v1",
        frame_idx=10,
        timestamp_ms=100,
        image_path="keyframes/v1/1.jpg",
        width=4,
        height=4,
    )
    pd.DataFrame([frame.model_dump(mode="python")]).to_parquet(
        metadata, index=False
    )
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "dense.index").write_bytes(b"index")
    (index_dir / "metadata.json").write_text(
        json.dumps({"vector_count": 1, "embedding_dim": 768}),
        encoding="utf-8",
    )
    pd.DataFrame([{"frame_id": "f1"}]).to_parquet(
        index_dir / "frame_mapping.parquet", index=False
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({
            "dataset": {"root": str(tmp_path), "frames_path": str(metadata)},
            "index": {"path": str(index_dir)},
            "inference": {"enabled": False},
        }),
        encoding="utf-8",
    )

    report = diagnose(config, sample_size=10)

    assert report["ready"] is True
    assert report["metadata"]["frames"] == 1
    assert report["frame_assets"]["available"] == 1
    assert report["visual_index"]["canonical_alignment"] is True
