"""HTTP endpoint helpers for offline-only model clients."""

from .endpoints import resolve_core_url, resolve_gpu_url

__all__ = ["resolve_core_url", "resolve_gpu_url"]
