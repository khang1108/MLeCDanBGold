"""Adapters cho các mô hình phân tích Video (Preprocessing).

Khởi tạo lười (Lazy initialization) cho các mô hình AI dùng để phát hiện ranh giới cảnh quay.

Các tính năng chính:
1. Shot Boundary Detection: Tích hợp mô hình TransNetV2 để tìm ranh giới các cú máy (shot).
2. Event Boundary Detection: Tích hợp mô hình GEBD để tìm các sự kiện chuyển động chính yếu.
3. Lazy Loading: Chỉ nạp tệp weights của mô hình vào GPU khi hàm xử lý video được gọi lần đầu."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.video import FrameMeta

EFFICIENTGEBD_OVERLAP = 20
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _module(name: str, path: Path) -> Any:
    """Import one source file without requiring an installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class TransNetDetector:
    """Return TransNetV2 shot-transition probabilities."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Keep the model unloaded until the first video."""
        self.config = config
        self.model: Any = None

    def _load(self) -> Any:
        """Load the official TensorFlow model once."""
        if self.model is not None:
            return self.model
        source = self.config.transnet_repo / "inference" / "transnetv2.py"
        weights = self.config.transnet_weights
        if not source.is_file() or not weights.exists():
            raise FileNotFoundError("TransNetV2 source or weights are unavailable")
        self.model = _module("hcmai_transnetv2", source).TransNetV2(
            str(weights)
        )
        return self.model

    def score(self, _path: Path, frames: np.ndarray) -> np.ndarray:
        """Return one score for each decoded frame."""
        scores, _ = self._load().predict_frames(frames)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class EfficientGEBDDetector:
    """Run the official EfficientGEBD checkpoint over temporal windows."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Keep the PyTorch model unloaded until the first video."""
        self.config = config
        self.model: Any = None
        self.model_config: Any = None
        self.positions: list[int] = []
        self.totals: list[float] = []
        self.counts: list[int] = []
        self.window: list[tuple[int, Any]] = []
        self.pending = 0
        self.next_ms = 0.0
        self.mean: Any = None
        self.std: Any = None

    def _load(self) -> tuple[Any, Any]:
        """Load the configured official model and checkpoint once."""
        if self.model is not None:
            return self.model, self.model_config
        repo = self.config.efficientgebd_repo
        cfg_path = self.config.efficientgebd_config
        checkpoint = self.config.efficientgebd_checkpoint
        if (
            not repo.is_dir()
            or not cfg_path.is_file()
            or not checkpoint.is_file()
        ):
            raise FileNotFoundError("EfficientGEBD source or checkpoint is unavailable")
        base = _module("hcmai_gebd_config", repo / "modeling" / "config.py")
        cfg = base._C.clone()
        cfg.merge_from_file(str(cfg_path))
        cfg.freeze()
        model_module = self._model_module(repo, cfg.MODEL.NAME)
        model_class = getattr(model_module, cfg.MODEL.NAME)
        model = self._without_pretraining(model_class, cfg)
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state), strict=True)
        self.model = model.to(self.config.device).eval()
        self.model_config = cfg
        return self.model, cfg

    @staticmethod
    def _model_module(repo: Path, name: str) -> Any:
        """Import the requested ResNet model without optional CSN packages."""
        package = "hcmai_efficientgebd_modeling"
        root = repo / "modeling"
        container = types.ModuleType(package)
        container.__path__ = [str(root)]
        sys.modules[package] = container
        backbone = types.ModuleType(f"{package}.backbone")
        unsupported = type("UnsupportedBackbone", (), {})
        for symbol in ("CSN", "CSNR50", "TSM", "VideoMAEv2"):
            setattr(backbone, symbol, unsupported)
        sys.modules[f"{package}.backbone"] = backbone
        sys.modules.setdefault("pickle5", pickle)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        filename = {
            "BaseModel": "baseline.py",
            "E2EModelDiff": "e2e_model_diff_former.py",
        }.get(name)
        if filename is None:
            raise ValueError(f"Unsupported EfficientGEBD model: {name}")
        return _module(f"{package}.{filename[:-3]}", root / filename)

    @staticmethod
    def _without_pretraining(model_class: Any, cfg: Any) -> Any:
        """Build from the supplied checkpoint without downloading ImageNet weights."""
        from torchvision import models

        name = cfg.MODEL.BACKBONE.NAME
        builder = getattr(models, name)

        def build(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("pretrained", None)
            return builder(*args, weights=None, **kwargs)

        setattr(models, name, build)
        try:
            return model_class(cfg)
        finally:
            setattr(models, name, builder)

    def _predict(self, images: list[Any], valid: int) -> np.ndarray:
        """Pad and score one fixed-length model window."""
        import torch
        model, cfg = self._load()
        length = int(cfg.INPUT.SEQUENCE_LENGTH)
        padded = images + [images[-1]] * (length - len(images))
        batch = torch.stack(padded).unsqueeze(0).to(self.config.device)
        with torch.inference_mode():
            scores = model({"imgs": batch})[0, -1, :valid]
        return scores.float().cpu().numpy()

    def start(self) -> None:
        """Reset streamed event detection for one video."""
        self.positions, self.totals, self.counts, self.window = [], [], [], []
        self.pending, self.next_ms = 0, 0.0
        import torch

        _, cfg = self._load()
        self.mean = torch.tensor(IMAGE_MEAN)[:, None, None]
        self.std = torch.tensor(IMAGE_STD)[:, None, None]
        if EFFICIENTGEBD_OVERLAP >= int(cfg.INPUT.SEQUENCE_LENGTH):
            raise ValueError("EfficientGEBD overlap must be shorter than its window")

    def update(self, frame: FrameMeta, source: Any) -> None:
        """Consume one decoded frame at the configured model rate."""
        if frame.timestamp_ms < self.next_ms:
            return
        import torch

        _, cfg = self._load()
        resolution = int(cfg.INPUT.RESOLUTION)
        image = source.to_image().convert("RGB").resize((resolution, resolution))
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        tensor = tensor.float() / 255
        self.positions.append(frame.decode_index)
        self.totals.append(0.0)
        self.counts.append(0)
        self.window.append((
            len(self.positions) - 1,
            (tensor - self.mean) / self.std,
        ))
        self.pending += 1
        length = int(cfg.INPUT.SEQUENCE_LENGTH)
        if len(self.window) == length:
            self._add_scores(self.window, self.totals, self.counts)
            self.window = self.window[
                length - EFFICIENTGEBD_OVERLAP:
            ]
            self.pending = 0
        interval = 1_000 / self.config.efficientgebd_sample_fps
        while self.next_ms <= frame.timestamp_ms:
            self.next_ms += interval

    def scores(self, frame_count: int) -> np.ndarray:
        """Finish the stream and interpolate scores to native frame positions."""
        if self.window and (self.pending or not any(self.counts)):
            self._add_scores(self.window, self.totals, self.counts)
        sampled = np.asarray(self.totals) / np.maximum(self.counts, 1)
        return np.interp(
            np.arange(frame_count), self.positions, sampled
        ).astype(np.float32)

    def _add_scores(
        self, window: list[tuple[int, Any]], totals: list[float], counts: list[int]
    ) -> None:
        """Accumulate one window while retaining only overlap tensors."""
        scores = self._predict([tensor for _, tensor in window], len(window))
        for (index, _), score in zip(window, scores):
            totals[index] += float(score)
            counts[index] += 1
