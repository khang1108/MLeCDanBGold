"""Strict and resumable visual embedding artifact tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from hcmai.common.config import EncoderConfig
from hcmai.retrieval.embedding.artifacts import EmbeddingArtifactBuilder

_SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import build_embeddings


class CountingEncoder:
    """Encode deterministic vectors while recording how many images were requested."""

    def __init__(self) -> None:
        self.config = EncoderConfig(batch_size=2)
        self.embedding_dim = 3
        self.image_count = 0

    def encode_images(self, images, stats=None) -> np.ndarray:
        """Return one stable vector per supplied image without model loading."""
        self.image_count += len(images)
        if stats is not None:
            stats.num_encoded += len(images)
            stats.embedding_dim = self.embedding_dim
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(images), 1))


@pytest.fixture
def frame_table(tmp_path: Path) -> Path:
    """Create canonical rows and their BTC-keyframe-shaped image paths."""
    root = tmp_path / "keyframes" / "L21_V001"
    root.mkdir(parents=True)
    for order in (1, 2, 3):
        Image.new("RGB", (8, 6)).save(root / f"{order:03d}.jpg")
    table = tmp_path / "frames.parquet"
    pd.DataFrame(
        [
            {
                "frame_id": f"frame-{order}",
                "video_id": "L21_V001",
                "frame_idx": order * 90,
                "timestamp_ms": order * 3_000,
                "keyframe_order": order,
                "image_path": f"L21_V001/{order:03d}.jpg",
            }
            for order in (1, 2, 3)
        ]
    ).to_parquet(table, index=False)
    return table


def _builder(
    tmp_path: Path,
    encoder: CountingEncoder,
    frame_table: Path,
    *,
    shard_size: int = 2,
) -> EmbeddingArtifactBuilder:
    """Create a strict, resumable builder over one deterministic fixture."""
    return EmbeddingArtifactBuilder(
        frames_path=frame_table,
        dataset_root=tmp_path / "keyframes",
        output_dir=tmp_path / "out",
        encoder_config=EncoderConfig(batch_size=2),
        encoder=encoder,
        strict=True,
        resume=True,
        shard_size=shard_size,
    )


def test_strict_visual_build_refuses_missing_image(
    tmp_path: Path, frame_table: Path
) -> None:
    """Strict builds refuse to publish a partial visual corpus."""
    (tmp_path / "keyframes" / "L21_V001" / "003.jpg").unlink()
    builder = _builder(tmp_path, CountingEncoder(), frame_table)

    with pytest.raises(RuntimeError, match="complete visual coverage"):
        builder.run()

    assert not builder.embeddings_file.exists()
    assert not builder.mapping_file.exists()


def test_visual_build_reuses_valid_completed_shard(
    tmp_path: Path, frame_table: Path
) -> None:
    """A second matching build must reuse completed canonical row slices."""
    encoder = CountingEncoder()
    first = _builder(tmp_path, encoder, frame_table)
    first.run()
    calls_after_first = encoder.image_count

    second = _builder(tmp_path, encoder, frame_table)
    second.run()

    assert encoder.image_count == calls_after_first


def test_visual_build_regenerates_mismatched_shard(
    tmp_path: Path, frame_table: Path
) -> None:
    """A shard with wrong canonical IDs is discarded instead of reused."""
    encoder = CountingEncoder()
    first = _builder(tmp_path, encoder, frame_table)
    first.run()
    calls_after_first = encoder.image_count
    shard = sorted((first.embeddings_dir / "shards").glob("*.npz"))[0]
    np.savez_compressed(
        shard,
        frame_ids=np.asarray(["wrong-1", "wrong-2"], dtype=str),
        vectors=np.ones((2, 3), dtype=np.float32),
    )

    second = _builder(tmp_path, encoder, frame_table)
    second.run()

    assert encoder.image_count == calls_after_first + 2


def test_visual_build_regenerates_corrupt_shard(
    tmp_path: Path, frame_table: Path
) -> None:
    """A malformed NPZ shard is rebuilt rather than aborting resume."""
    encoder = CountingEncoder()
    first = _builder(tmp_path, encoder, frame_table)
    first.run()
    calls_after_first = encoder.image_count
    shard = sorted((first.embeddings_dir / "shards").glob("*.npz"))[0]
    shard.write_bytes(b"truncated checkpoint")

    second = _builder(tmp_path, encoder, frame_table)
    second.run()

    assert encoder.image_count == calls_after_first + 2


def test_cli_failed_build_retains_checkpoints_for_repair_resume(
    tmp_path: Path,
    frame_table: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired CLI build reuses shards completed before strict failure."""
    output_dir = tmp_path / "published"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  version: fixture-v1",
                f"  frames_path: {frame_table}",
                f"  root: {tmp_path / 'keyframes'}",
                "index:",
                f"  path: {output_dir / 'indexes' / 'visual'}",
            ]
        )
    )
    args = SimpleNamespace(
        config=str(config_path),
        frames=str(frame_table),
        dataset_root=str(tmp_path / "keyframes"),
        output=str(output_dir),
        model_config="unused.yaml",
        strict=True,
        resume=True,
        shard_size=2,
    )
    encoder = CountingEncoder()
    monkeypatch.setattr(
        build_embeddings.LLMServiceConfig,
        "from_yaml",
        lambda _path: SimpleNamespace(visual_embedding=EncoderConfig(batch_size=2)),
    )
    monkeypatch.setattr(
        build_embeddings,
        "EmbeddingArtifactBuilder",
        lambda **kwargs: EmbeddingArtifactBuilder(encoder=encoder, **kwargs),
    )
    (tmp_path / "keyframes" / "L21_V001" / "003.jpg").unlink()

    with pytest.raises(RuntimeError, match="complete visual coverage"):
        build_embeddings._run(args)

    checkpoint_dir = output_dir.with_name(".published.visual-checkpoints")
    assert sorted(checkpoint_dir.glob("*.npz"))
    assert encoder.image_count == 2
    assert not output_dir.exists()
    Image.new("RGB", (8, 6)).save(tmp_path / "keyframes" / "L21_V001" / "003.jpg")

    build_embeddings._run(args)

    assert encoder.image_count == 3
    assert (output_dir / "indexes" / "visual" / "dense.index").is_file()
