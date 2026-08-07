"""Lazy adapters for the official boundary-detection models."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from hcmai.data.preprocessing.config import PreprocessingConfig
from hcmai.data.preprocessing.video import iter_source_frames


def _module(name: str, path: Path) -> Any:
    """Import one source file without requiring an installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
        repo = self.config.transnet_repo
        source = repo / "inference" / "transnetv2.py" if repo else Path()
        if not repo or not source.is_file():
            raise FileNotFoundError("TransNetV2 checkout is not configured")
        weights = self.config.transnet_weights
        self.model = _module("hcmai_transnetv2", source).TransNetV2(
            str(weights) if weights else None
        )
        return self.model

    def score(self, _path: Path, frames: np.ndarray) -> np.ndarray:
        """Return one score for each decoded frame."""
        if not self.config.transnet_enabled:
            return np.zeros(len(frames), dtype=np.float32)
        scores, _ = self._load().predict_frames(frames)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class EfficientGEBDDetector:
    """Run the official EfficientGEBD checkpoint over temporal windows."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Keep the PyTorch model unloaded until it is enabled."""
        self.config = config
        self.model: Any = None
        self.model_config: Any = None

    def _load(self) -> tuple[Any, Any]:
        """Load the configured official model and checkpoint once."""
        if self.model is not None:
            return self.model, self.model_config
        repo = self.config.efficientgebd_repo
        cfg_path = self.config.efficientgebd_config
        checkpoint = self.config.efficientgebd_checkpoint
        if not repo or not cfg_path or not checkpoint:
            raise FileNotFoundError("EfficientGEBD paths are not configured")
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
        self.model = model.to(self.config.efficientgebd_device).eval()
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
        images += [images[-1]] * (length - len(images))
        batch = torch.stack(images).unsqueeze(0).to(self.config.efficientgebd_device)
        with torch.inference_mode():
            scores = model({"imgs": batch})[0, -1, :valid]
        return scores.float().cpu().numpy()

    def score(self, path: Path, frames: np.ndarray) -> np.ndarray:
        """Return interpolated event-boundary scores for every decoded frame."""
        if not self.config.efficientgebd_enabled:
            return np.zeros(len(frames), dtype=np.float32)
        _, cfg = self._load()
        length = int(cfg.INPUT.SEQUENCE_LENGTH)
        overlap = self.config.efficientgebd_overlap_frames
        if overlap >= length:
            raise ValueError("EfficientGEBD overlap must be shorter than its window")
        step = length - overlap
        positions: list[int] = []
        totals: list[float] = []
        counts: list[int] = []
        window: list[tuple[int, Any]] = []
        pending = 0
        for position, tensor in self._samples(path, int(cfg.INPUT.RESOLUTION)):
            positions.append(position)
            totals.append(0.0)
            counts.append(0)
            window.append((len(positions) - 1, tensor))
            pending += 1
            if len(window) == length:
                self._add_scores(window, totals, counts)
                window = window[step:]
                pending = 0
        if window and (pending or not any(counts)):
            self._add_scores(window, totals, counts)
        sampled = np.asarray(totals) / np.maximum(counts, 1)
        return np.interp(np.arange(len(frames)), positions, sampled).astype(np.float32)

    def _add_scores(
        self, window: list[tuple[int, Any]], totals: list[float], counts: list[int]
    ) -> None:
        """Accumulate one window while retaining only overlap tensors."""
        scores = self._predict([tensor for _, tensor in window], len(window))
        for (index, _), score in zip(window, scores):
            totals[index] += float(score)
            counts[index] += 1

    def _samples(self, path: Path, resolution: int) -> Iterator[tuple[int, Any]]:
        """Yield normalized RGB samples at the configured model rate."""
        import torch

        interval = 1_000 / self.config.efficientgebd_sample_fps
        next_ms = 0.0
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        for record, frame in iter_source_frames(path):
            if record.timestamp_ms < next_ms:
                continue
            image = frame.to_image().convert("RGB").resize((resolution, resolution))
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
            tensor = tensor.float() / 255
            yield record.decode_index, (tensor - mean) / std
            while next_ms <= record.timestamp_ms:
                next_ms += interval
