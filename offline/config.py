"""Small configuration boundary for local V3C/VBS preparation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class VBSPathsConfig:
    videos_root: Path
    work_root: Path
    frames_root: Path
    artifacts_root: Path
    indexes_root: Path
    state_root: Path


@dataclass(frozen=True, slots=True)
class InferenceEndpointConfig:
    base_url: str
    timeout_seconds: float
    max_concurrency: int
    endpoints: dict[str, str]

    def url(self, capability: str) -> str:
        try:
            path = self.endpoints[capability]
        except KeyError as error:
            raise KeyError(f"Inference capability is not configured: {capability}") from error
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True, slots=True)
class VBSConfig:
    paths: VBSPathsConfig
    core: InferenceEndpointConfig
    gpu: InferenceEndpointConfig

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/vbs_prepare.yaml") -> "VBSConfig":
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"VBS config must contain a mapping: {config_path}")
        project_root = config_path.parent.parent
        paths = raw.get("paths")
        inference = raw.get("inference")
        if not isinstance(paths, dict) or not isinstance(inference, dict):
            raise ValueError("VBS config requires top-level paths and inference mappings")

        def resolved(name: str, default: str | None = None) -> Path:
            value = paths.get(name, default)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"paths.{name} must be a non-empty path")
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()

        work = resolved("work_root", "data")
        path_config = VBSPathsConfig(
            videos_root=resolved("videos_root"),
            work_root=work,
            frames_root=resolved("frames_root", "data/frames"),
            artifacts_root=resolved("artifacts_root", "data/artifacts"),
            indexes_root=resolved("indexes_root", "data/indexes"),
            state_root=resolved("state_root", "data/state"),
        )
        return cls(
            paths=path_config,
            core=_endpoint(inference, "core", "VBS_CORE_API_URL"),
            gpu=_endpoint(inference, "gpu", "VBS_GPU_API_URL"),
        )


def _endpoint(raw: dict[str, Any], name: str, env_name: str) -> InferenceEndpointConfig:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"inference.{name} must be a mapping")
    configured_url = value.get("base_url")
    base_url = os.getenv(env_name, str(configured_url or "")).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"inference.{name}.base_url must be HTTP(S)")
    endpoints = value.get("endpoints", {})
    if not isinstance(endpoints, dict) or not endpoints:
        raise ValueError(f"inference.{name}.endpoints must be a non-empty mapping")
    return InferenceEndpointConfig(
        base_url=base_url,
        timeout_seconds=float(value.get("timeout_seconds", 120)),
        max_concurrency=int(value.get("max_concurrency", 1)),
        endpoints={str(key): str(path) for key, path in endpoints.items()},
    )


__all__ = ["InferenceEndpointConfig", "VBSConfig", "VBSPathsConfig"]
