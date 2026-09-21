"""Resolve capability-specific model API roots for offline preparation."""

from __future__ import annotations

import os
from pathlib import Path

from offline.config import VBSConfig


def _from_vbs_config(config_path: str | Path, kind: str) -> str | None:
    try:
        config = VBSConfig.from_yaml(config_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    endpoint = config.core if kind == "core" else config.gpu
    return endpoint.base_url


def resolve_core_url(config_path: str | Path, fallback: str | None = None) -> str:
    """Return the embedding/text API root, with environment override."""
    value = os.getenv("VBS_CORE_API_URL") or _from_vbs_config(config_path, "core") or fallback
    if not value:
        raise ValueError("Core inference URL is not configured")
    return value.rstrip("/")


def resolve_gpu_url(config_path: str | Path, fallback: str | None = None) -> str:
    """Return the GPU-heavy API root, with environment override."""
    value = os.getenv("VBS_GPU_API_URL") or _from_vbs_config(config_path, "gpu") or fallback
    if not value:
        raise ValueError("GPU inference URL is not configured")
    return value.rstrip("/")
