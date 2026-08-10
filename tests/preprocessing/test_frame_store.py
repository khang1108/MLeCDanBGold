"""Smoke tests for the compact adaptive FrameStore builder."""

from pathlib import Path

import av
import numpy as np

from hcmai.common.config import AppConfig
from hcmai.data import FrameStore
from hcmai.data.pipeline import DataService
from hcmai.data.preprocessing import PreprocessingConfig, prepare_frame_store
from hcmai.data.preprocessing.selection import deduplicate, select_candidates
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
    }
    values.update(updates)
    return PreprocessingConfig.model_validate(values)


def _video(path: Path) -> None:
    """Write a tiny ten-FPS video fixture."""

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
    retained = deduplicate(candidates, embeddings, config)
    assert len(retained) < len(candidates)
    assert any(item.frame.frame_idx == 2 for item in retained)


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
    assert settings.dataset.root == Path("artifacts/frame_store")
    assert settings.dataset.frames_path == Path(
        "artifacts/frame_store/frames.parquet"
    )


def test_prepare_publishes_one_resumable_frame_store(
    tmp_path: Path, monkeypatch,
) -> None:
    """Publish only images and metadata, then reuse the completed checkpoint."""

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
    store = FrameStore.load(output)
    data = DataService.load(output, dataset_root=config.output_root)
    assert tuple(path.name for path in config.output_root.iterdir()) == (
        "images", "frames.parquet",
    )
    assert not partial.exists()
    assert all((config.output_root / frame.image_path).is_file()
               for frame in store.iter_frames())
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
