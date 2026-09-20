"""Ensure the context-index entry point preserves canonical dataset lineage."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hcmai.common.config import EncoderConfig
from offline.enrichment.context.models import FrameContext
from scripts.indexing import build_retrieval_indexes as cli


@pytest.mark.parametrize("context_lineage", ["v3c-v1", "other-corpus"])
def test_context_index_uses_canonical_manifest(tmp_path, monkeypatch, context_lineage):
    """Reuse a projection without a manifest but reject different source lineage."""
    source = tmp_path / "source"
    source.mkdir()
    frames_path = source / "frames.parquet"
    row = dict(frame_id="f1", video_id="v1", frame_idx=100,
               timestamp_ms=4000, image_path="frames/f1.jpg")
    pd.DataFrame([row]).to_parquet(frames_path)
    (source / "manifest.json").write_text(json.dumps({"frame_store_id": "v3c-v1"}))
    projected = tmp_path / "projected_frames.parquet"
    pd.DataFrame([row]).to_parquet(projected)

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    context_path = context_dir / "frame_context_v1.parquet"
    context = FrameContext(
        **{key: row[key] for key in ("frame_id", "video_id", "frame_idx", "timestamp_ms")},
        context_text="A cyclist", caption_text="A cyclist", caption_available=True,
        context_version="ctx-v1", caption_version="cap-v1", ocr_version="none",
        object_version="none", frame_store_id=context_lineage,
    )
    pd.DataFrame([context.model_dump()]).to_parquet(context_path)
    (context_dir / "manifest.json").write_text(json.dumps({
        "frame_store_id": context_lineage, "context_version": "ctx-v1",
    }))
    config = SimpleNamespace(
        dataset=SimpleNamespace(frames_path=frames_path, context_path=context_path, version="v1"),
        indexes=SimpleNamespace(context=tmp_path / "index"),
    )
    encoder = SimpleNamespace(
        config=EncoderConfig(),
        encode_text=lambda texts: np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1)),
    )
    monkeypatch.setattr(cli, "_config_fingerprint", lambda config: "test")
    monkeypatch.setattr(cli, "_stamp_config_fingerprint", lambda index, *args: index)
    if context_lineage != "v3c-v1":
        with pytest.raises(ValueError, match="lineage mismatch"):
            cli.build_context(config, None, projected, encoder=encoder)
        return

    index = cli.build_context(config, None, projected, encoder=encoder)
    assert index.mapping["frame_id"].tolist() == ["f1"]
    assert index.mapping["frame_idx"].tolist() == [100]
