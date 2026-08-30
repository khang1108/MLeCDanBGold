"""Public deterministic FrameContext V1 configuration and build interfaces."""

from .config import FrameContextConfig
from .serializer import serialize_frame_context


def build_frame_context(*args, **kwargs):
    """Build and publish FrameContext after loading its optional dependencies."""

    from .builder import build_frame_context as _build_frame_context

    return _build_frame_context(*args, **kwargs)

__all__ = [
    "FrameContextConfig",
    "build_frame_context",
    "serialize_frame_context",
]
