"""Smoke tests for the compact adaptive FrameStore builder."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

try:
    import av
except ModuleNotFoundError:
    av = None

from hcmai.common.config import AppConfig
from hcmai.common.schemas.frame import FrameRecord
from hcmai.data.pipeline import DataService
from hcmai.data.preprocessing import PreprocessingConfig, prepare_frame_store
from hcmai.data.preprocessing.prepare import (
    _config_hash,
    _publish_directory,
    _record,
)
from hcmai.data.preprocessing.selection import (
    CandidateFrame,
    deduplicate,
    restore_maximum_gap,
    select_candidates,
)
from hcmai.data.preprocessing.video import FrameMeta, iter_source_frames


class FakeDetector:
    def start(self) -> None:
        """Start one fake streamed detection pass."""

    def update(self, _frame: FrameMeta, _source: object) -> None:
        """Accept one decoded frame without retaining its image."""

    def scores(self, frame_count: int) -> np.ndarray:
        """Mark the middle frame as an event boundary."""

        scores = np.zeros(frame_count, dtype=np.float32)
        scores[len(scores) // 2] = 1.0
        return scores

    def score(self, _path: Path, frames: np.ndarray) -> np.ndarray:
        """Mark the middle frame as a boundary."""

        scores = np.zeros(len(frames), dtype=np.float32)
        scores[len(scores) // 2] = 1.0
        return scores


class FakeEncoder:
    def encode(self, images: list[object]) -> np.ndarray:
        """Produce one constant embedding per image."""

        return np.ones((len(images), 2), dtype=np.float32) / np.sqrt(2)


def _config(root: Path, **updates: object) -> PreprocessingConfig:
    """Create the smallest full-stack test configuration."""

    values = {
        "videos_root": root,
        "output_root": root / "frame_store",
        "transnet_repo": root / "TransNetV2",
        "transnet_weights": root / "transnetv2-weights",
        "efficientgebd_repo": root / "EfficientGEBD",
        "efficientgebd_config": root / "efficientgebd.yaml",
        "efficientgebd_checkpoint": root / "efficientgebd.pth",
        "dino_revision": "test-revision",
    }
    values.update(updates)
    return PreprocessingConfig.model_validate(values)


def _video(path: Path) -> None:
    """Write a tiny ten-FPS video fixture."""

    if av is None:
        pytest.skip("PyAV is installed by the preprocessing extra")
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(10):
            rgb = np.full((48, 64, 3), index * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_candidate_union_gap_and_safe_dedup(tmp_path: Path) -> None:
    """Keep hard triggers and drop only an unprotected same-shot duplicate."""

    config = _config(
        tmp_path,
        maximum_gap_ms=250,
        minimum_gap_ms=250,
    )
    frames = [FrameMeta("V", i, i, i, "1/10", i * 100, 8, 8) for i in range(6)]
    shots = np.array([0, 0, 1, 1, 0, 0], dtype=np.float32)
    candidates = select_candidates(frames, shots, np.zeros(6), config)
    assert sum("shot_boundary" in item.reasons for item in candidates) == 1
    assert all(
        right.frame.timestamp_ms - left.frame.timestamp_ms <= 250
        for left, right in zip(candidates, candidates[1:])
    )
    embeddings = np.ones((len(candidates), 2), dtype=np.float32) / np.sqrt(2)
    image_path = tmp_path / "unchanged.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)
    retained = deduplicate(
        candidates,
        embeddings,
        [image_path] * len(candidates),
        config,
    )
    assert len(retained) < len(candidates)
    assert any(item.frame.frame_idx == 2 for item in retained)


def test_dedup_repairs_the_hard_maximum_gap(tmp_path: Path) -> None:
    """Reinsert coverage candidates removed by semantic deduplication."""

    config = _config(
        tmp_path,
        maximum_gap_ms=1_000,
        minimum_gap_ms=1_000,
    )
    frames = [
        FrameMeta("V", index, index, index, "1/10", index * 100, 8, 8)
        for index in range(31)
    ]
    shots = np.zeros(31, dtype=np.float32)
    shots[10] = 1.0
    candidates = select_candidates(frames, shots, np.zeros(31), config)
    embeddings = np.ones((len(candidates), 2), dtype=np.float32) / np.sqrt(2)
    image_path = tmp_path / "unchanged.jpg"
    Image.new("RGB", (8, 8), "white").save(image_path)

    retained = restore_maximum_gap(
        candidates,
        deduplicate(
            candidates,
            embeddings,
            [image_path] * len(candidates),
            config,
        ),
        config,
    )

    assert all(
        right.frame.timestamp_ms - left.frame.timestamp_ms <= 1_000
        for left, right in zip(retained, retained[1:])
    )


def test_internal_identity_does_not_depend_on_submission_frame_idx() -> None:
    """Allow decoded frames to share a submission mapping without collision."""

    first = CandidateFrame(
        frame=FrameMeta("V", 7, 3, 7, "1/10", 300, 8, 8),
        shot_id=0,
        event_id=0,
        shot_score=0.0,
        event_score=0.0,
        reasons=("coverage_anchor",),
        protected=True,
    )
    second = CandidateFrame(
        frame=FrameMeta("V", 8, 3, 8, "1/10", 350, 8, 8),
        shot_id=0,
        event_id=0,
        shot_score=0.0,
        event_score=0.0,
        reasons=("coverage_anchor",),
        protected=True,
    )

    rows = [_record(first, Path("7.jpg")), _record(second, Path("8.jpg"))]

    assert rows[0]["frame_idx"] == rows[1]["frame_idx"] == 3
    assert rows[0]["frame_id"] != rows[1]["frame_id"]


def test_directory_publication_restores_previous_images_on_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    """Keep the previous complete image set if final publication fails."""

    final_dir = tmp_path / "video"
    partial_dir = tmp_path / ".video.partial"
    final_dir.mkdir()
    partial_dir.mkdir()
    (final_dir / "old.jpg").touch()
    (partial_dir / "new.jpg").touch()
    original_replace = Path.replace

    def failing_replace(path: Path, target: Path) -> Path:
        if path == partial_dir:
            raise OSError("simulated publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated publication failure"):
        _publish_directory(partial_dir, final_dir)

    assert (final_dir / "old.jpg").is_file()
    assert not (final_dir / "new.jpg").exists()


def test_frame_index_uses_zero_based_milliseconds_and_fps(tmp_path: Path) -> None:
    """Map elapsed milliseconds to the nearest native frame index."""

    path = tmp_path / "L21_V001.mp4"
    _video(path)
    frames = [record for record, _ in iter_source_frames(path)]
    assert (frames[0].timestamp_ms, frames[0].frame_idx) == (0, 0)
    assert [frame.frame_idx for frame in frames] == [
        round(frame.timestamp_ms * 10 / 1_000) for frame in frames
    ]


def test_minimal_yaml_reads_gpu_overrides(tmp_path: Path, monkeypatch) -> None:
    """Load the compact YAML and only override deployment settings."""

    path = tmp_path / "preprocessing.yaml"
    path.write_text(
        """preprocessing:
  videos_root: videos
  output_root: artifacts/frame_store
  transnet_repo: models/TransNetV2
  transnet_weights: models/TransNetV2/weights
  efficientgebd_repo: models/EfficientGEBD
  efficientgebd_config: models/EfficientGEBD/model.yaml
  efficientgebd_checkpoint: models/EfficientGEBD/model.pth
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HCMAI_PREPROCESSING_DEVICE", "cpu")
    monkeypatch.setenv("HCMAI_PREPROCESSING_DINO_DTYPE", "float32")

    config = PreprocessingConfig.from_yaml(path)

    assert config.device == "cpu"
    assert config.dino_dtype == "float32"
    assert not hasattr(config, "dino_enabled")
    assert not hasattr(config, "resume")


def test_runtime_paths_target_public_frame_store() -> None:
    """Use canonical metadata and its image root, not vector mappings."""

    root = Path(__file__).resolve().parents[2]
    settings = AppConfig.from_yaml(root / "configs/baseline.yaml")
    assert settings.dataset.root == Path("data")
    assert settings.dataset.frames_path == Path(
        "artifacts/frame_store/frames.parquet"
    )


def test_prepare_publishes_one_resumable_frame_store(
    tmp_path: Path, monkeypatch,
) -> None:
    """Publish only images and metadata, then reuse the completed checkpoint."""

    if av is None:
        pytest.skip("PyAV is installed by the preprocessing extra")
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos / "L21_V001.mp4")
    source_open = av.open
    decode_count = 0

    def counted_open(*args, **kwargs):
        """Count source-video decode passes."""
        nonlocal decode_count
        if not kwargs.get("mode") and len(args) == 1:
            decode_count += 1
        return source_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", counted_open)
    config = _config(
        tmp_path,
        videos_root=videos,
        output_root=tmp_path / "frame_store",
        maximum_gap_ms=400,
        minimum_gap_ms=200,
    )
    partial = config.output_root / "images/L21/.L21_V001.partial"
    partial.mkdir(parents=True)
    (partial / "junk.jpg").touch()
    detector, encoder = FakeDetector(), FakeEncoder()
    output = prepare_frame_store(
        config, shot_detector=detector, event_detector=detector, encoder=encoder
    )
    data = DataService.load(output, dataset_root=config.output_root)
    assert tuple(sorted(path.name for path in config.output_root.iterdir())) == (
        "frames.parquet", "images", "manifest.json",
    )
    assert not partial.exists()
    assert all((config.output_root / frame.image_path).is_file()
               for frame in data.iter_frames())
    assert all(data.resolve_frame_asset(frame).is_file()
               for frame in data.iter_frames())
    assert decode_count == 2
    prepare_frame_store(
        config, shot_detector=detector, event_detector=detector, encoder=encoder
    )
    assert decode_count == 2
    prepare_frame_store(
        config,
        shot_detector=detector,
        event_detector=detector,
        encoder=encoder,
        resume=False,
    )
    assert decode_count == 4


def test_limit_uses_an_isolated_frame_store(tmp_path: Path, monkeypatch) -> None:
    """Never truncate the configured full-corpus artifact during a smoke run."""

    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("L21_V001.mp4", "L21_V002.mp4"):
        (videos / name).touch()
    config = _config(tmp_path, videos_root=videos)

    def fake_prepare(path: Path, *_args, **_kwargs) -> pd.DataFrame:
        candidate = CandidateFrame(
            frame=FrameMeta(path.stem, 0, 0, 0, "1/10", 0, 8, 8),
            shot_id=0,
            event_id=0,
            shot_score=0.0,
            event_score=0.0,
            reasons=("coverage_anchor",),
            protected=True,
        )
        return pd.DataFrame(
            [_record(candidate, Path("images") / f"{path.stem}.jpg")],
            columns=list(FrameRecord.model_fields),
        )

    monkeypatch.setattr(
        "hcmai.data.preprocessing.prepare._prepare_video", fake_prepare
    )
    full = prepare_frame_store(
        config, shot_detector=object(), event_detector=object(), encoder=object()
    )
    limited = prepare_frame_store(
        config,
        shot_detector=object(),
        event_detector=object(),
        encoder=object(),
        limit=1,
    )

    assert full == config.output_root / "frames.parquet"
    assert limited == tmp_path / "frame_store.limit-1/frames.parquet"
    assert len(pd.read_parquet(full)) == 2
    assert len(pd.read_parquet(limited)) == 1


def test_config_hash_tracks_checkpoint_contents(tmp_path: Path) -> None:
    """Invalidate resume when a model file changes at the same path."""

    checkpoint = tmp_path / "efficientgebd.pth"
    checkpoint.write_bytes(b"first-version")
    config = _config(tmp_path, efficientgebd_checkpoint=checkpoint)

    first = _config_hash(config)
    checkpoint.write_bytes(b"other-version")

    assert _config_hash(config) != first
